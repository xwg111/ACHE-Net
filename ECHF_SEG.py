import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ChannelShuffle(nn.Module):
    def __init__(self, num_groups):
        super(ChannelShuffle, self).__init__()
        self.num_groups = num_groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.num_groups

        # Reshape
        x = x.view(batch_size, self.num_groups, channels_per_group, height, width)

        # Transpose
        x = torch.transpose(x, 1, 2).contiguous()

        # Flatten
        x = x.view(batch_size, -1, height, width)

        return x

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        x = F.pad(x, (0, pad_w, 0, pad_h))

        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2

        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return torch.cat((x_LL, x_HL, x_LH, x_HH), dim=1)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
        h = torch.zeros([out_batch, out_channel, out_height, out_width], dtype=torch.float32, device=x.device)
        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
        return h

class DHF(nn.Module):
    def __init__(self, in_channels, reduction=16, max_wavelet_level=3):
        super().__init__()
        self.in_channels = in_channels
        self.max_wavelet_level = max_wavelet_level
        self.dwt = DWT()
        self.iwt = IWT()

        # 深度可分离卷积处理高频分量（3C输入，3C输出）
        self.ds_conv = nn.Conv2d(3 * in_channels, 3 * in_channels, kernel_size=3,
                      padding=1, groups=3 * in_channels)

        # 通道调整层
        self.conv_adjust = nn.Conv2d(
            max_wavelet_level * in_channels,  # 每个层级输出in_channels
            in_channels,
            kernel_size=1
        )
        self.sigmoid = nn.Sigmoid()

    def wavelet_decomposition(self, x):
        """保存各层级的LL、LH、HL、HH"""
        coeffs = []
        current = x
        level = 0
        while level < self.max_wavelet_level:
            if current.size(2) >= 2 and current.size(3) >= 2:
                decomposed = self.dwt(current)
                ll, lh, hl, hh = torch.split(decomposed, decomposed.size(1)//4, dim=1)
                coeffs.append((ll, lh, hl, hh))
                current = ll
                level += 1
            else:
                break
        return coeffs

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 自适应小波分解
        coeffs = self.wavelet_decomposition(x)
        reconstructed_features = []

        # 2. 逐层处理分解结果
        for level_coeff in coeffs:
            ll, lh, hl, hh = level_coeff

            # 处理高频分量
            highs = torch.cat([lh, hl, hh], dim=1)  # [B, 3*C, H, W]
            processed_highs = self.ds_conv(highs)    # 深度可分离卷积
            lh_p, hl_p, hh_p = torch.split(processed_highs, self.in_channels, dim=1)

            # 逆小波变换
            combined = torch.cat([ll, lh_p, hl_p, hh_p], dim=1)
            iwt_feature = self.iwt(combined)

            # 上采样到原始尺寸
            iwt_feature = F.interpolate(iwt_feature, (H, W),
                                        mode='bilinear', align_corners=False)
            reconstructed_features.append(iwt_feature)

        # 3. 合并特征生成注意力图
        if reconstructed_features:
            merged = torch.cat(reconstructed_features, dim=1)  # [B, L*C, H, W]
            merged = self.conv_adjust(merged)
        else:
            merged = x  # 无分解时直接使用原特征

        att = self.sigmoid(merged)
        return x * att

class WMFE(torch.nn.Module):
    def __init__(self, n_filts, out_channels,  max_inv_fctr=3):
        super().__init__()

        def compute_k(channels):
            max_k = 7
            min_k = 3
            scale_factor = 128

            k = max_k - 2 * (channels // scale_factor)
            k = max(min(k, max_k), min_k)
            return k + 1 if k % 2 == 0 else k

        self.k = compute_k(n_filts)
        self.a = 3 - 2 * (self.k != 3)
        self.inv_fctr = self._calculate_inv_fctr(n_filts, max_inv_fctr)

        expanded_channels = n_filts * self.inv_fctr
        self.conv1 = nn.Conv2d(n_filts, expanded_channels, kernel_size=1,stride=1,padding=0)
        self.norm1 = nn.BatchNorm2d(expanded_channels)

        self.conv2 = nn.Conv2d(
            expanded_channels,
            expanded_channels,
            kernel_size=3,
            padding=1,
            groups=expanded_channels
        )
        self.conv5 = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=(self.a, self.k),
                               padding=(self.a // 2, self.k // 2), groups=expanded_channels)
        self.conv6 = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=(self.k, self.a),
                               padding=(self.k // 2, self.a // 2), groups=expanded_channels)
        self.conv7 = nn.Conv2d(expanded_channels * 2, expanded_channels, kernel_size=1)
        self.cs = ChannelShuffle(expanded_channels)
        self.norm2 = nn.GroupNorm(num_groups=expanded_channels,num_channels=expanded_channels)

        self.conv3 = nn.Conv2d(expanded_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(p=0.1)
        self.norm3 = nn.BatchNorm2d(out_channels)

        self.activation = nn.GELU()

    @staticmethod
    def _calculate_inv_fctr(n_filts, max_inv_fctr):
        base_channels = 32
        inv_fctr = min(max_inv_fctr, max(1, int(max_inv_fctr - 0.5 * (n_filts // base_channels))))
        return inv_fctr

    def forward(self, inp):
        x = self.conv1(inp)
        x0 = x
        x = self.norm1(x)
        x = self.activation(x)

        x1 = self.conv5(x)
        x2 = self.conv6(x)
        x = self.conv7(torch.cat([x1, x2], dim=1))
        x = self.conv2(x)
        x = self.cs(x)
        x = self.norm2(x)
        x += x0
        x = self.activation(x)

        x = self.conv3(x)
        x = self.dropout(x)
        x = self.norm3(x)
        x = self.activation(x)
        return x

class FrequencyFusion(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        self.dwt = DWT()
        self.iwt = IWT()

        self.channel_compression = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels // reduction, 1),  # 修改这里
            nn.BatchNorm2d(in_channels // reduction),
            nn.SiLU()
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels // reduction, in_channels * 4, 1),  # 修改这里
            nn.Sigmoid()
        )

        self.feature_rebuild = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

    def forward(self, x_enc: Tensor, x_dec: Tensor) -> Tensor:
        B, C, H, W = x_dec.shape

        enc_coeffs = self.dwt(x_enc)
        enc_ll, enc_lh, enc_hl, enc_hh = torch.split(enc_coeffs, enc_coeffs.size(1) // 4, dim=1)

        dec_coeffs = self.dwt(x_dec)
        dec_ll, dec_lh, dec_hl, dec_hh = torch.split(dec_coeffs, dec_coeffs.size(1) // 4, dim=1)

        fused_ll = enc_ll + dec_ll  # [B, C, H, W]
        fused_lh = enc_lh + dec_lh  # [B, C, H, W]
        fused_hl = enc_hl + dec_hl  # [B, C, H, W]
        fused_hh = enc_hh + dec_hh  # [B, C, H, W]

        compressed = self.channel_compression(
            torch.cat([fused_ll, fused_lh, fused_hl, fused_hh], dim=1)
        )  # [B, C//R, H, W]

        att_all = self.attention(compressed)  # [B, 4C, 1, 1]

        att_ll = att_all[:, :C, ...]  # [B, C, 1, 1]
        att_lh = att_all[:, C:2 * C, ...]  # [B, C, 1, 1]
        att_hl = att_all[:, 2 * C:3 * C, ...]  # [B, C, 1, 1]
        att_hh = att_all[:, 3 * C:4 * C, ...]  # [B, C, 1, 1]

        weighted_ll = fused_ll * att_ll.expand_as(fused_ll)
        weighted_lh = fused_lh * att_lh.expand_as(fused_lh)
        weighted_hl = fused_hl * att_hl.expand_as(fused_hl)
        weighted_hh = fused_hh * att_hh.expand_as(fused_hh)

        rebuilt = self.iwt(
            torch.cat([weighted_ll, weighted_lh, weighted_hl, weighted_hh], dim=1))

        return self.feature_rebuild(torch.cat([x_dec, rebuilt], dim=1))

class ECHF(torch.nn.Module):

    def __init__(self, n_channels, n_classes, n_filts=32):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.pool = torch.nn.MaxPool2d(2)

        self.cnv11 = WMFE(n_channels, n_filts)
        self.att1 = DHF(in_channels=n_filts,max_wavelet_level=3)

        self.cnv21 = WMFE(n_filts, n_filts * 2)
        self.att2 = DHF(in_channels=n_filts * 2,max_wavelet_level=3)

        self.cnv31 = WMFE(n_filts * 2, n_filts * 4)
        self.att3 = DHF(in_channels=n_filts * 4, max_wavelet_level=3)

        self.cnv41 = WMFE(n_filts * 4, n_filts * 8)
        self.att4 = DHF(in_channels=n_filts * 8, max_wavelet_level=2)

        self.cnv51 = WMFE(n_filts * 8, n_filts * 16)
        self.att5 = DHF(in_channels=n_filts * 16, max_wavelet_level=1)

        self.fusion1 = FrequencyFusion(n_filts * 8)
        self.fusion2 = FrequencyFusion(n_filts * 4)
        self.fusion3 = FrequencyFusion(n_filts * 2)
        self.fusion4 = FrequencyFusion(n_filts)


        self.up6 = torch.nn.ConvTranspose2d(n_filts * 16, n_filts * 8, kernel_size=(2, 2), stride=2)
        self.cnv61 = WMFE(n_filts * 8 + n_filts * 8, n_filts * 8)
        self.att6 = DHF(in_channels=n_filts * 8, max_wavelet_level=2)

        self.up7 = torch.nn.ConvTranspose2d(n_filts * 8, n_filts * 4, kernel_size=(2, 2), stride=2)
        self.cnv71 = WMFE(n_filts * 4 + n_filts * 4, n_filts * 4)
        self.att7 = DHF(in_channels=n_filts * 4, max_wavelet_level=3)

        self.up8 = torch.nn.ConvTranspose2d(n_filts * 4, n_filts * 2, kernel_size=(2, 2), stride=2)
        self.cnv81 = WMFE(n_filts * 2 + n_filts * 2, n_filts * 2)
        self.att8 = DHF(in_channels=n_filts * 2, max_wavelet_level=3)

        self.up9 = torch.nn.ConvTranspose2d(n_filts * 2, n_filts, kernel_size=(2, 2), stride=2)
        self.cnv91 = WMFE(n_filts + n_filts, n_filts)
        self.att9 = DHF(in_channels=n_filts, max_wavelet_level=3)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        if n_classes == 1:
            self.out = torch.nn.Conv2d(n_filts, n_classes, kernel_size=(1, 1))
            self.last_activation = torch.nn.Sigmoid()
        else:
            self.out = torch.nn.Conv2d(n_filts, n_classes + 1, kernel_size=(1, 1))
            self.last_activation = None

    def forward(self, x):

        x1 = x
        x2 = self.cnv11(x1)
        x2 = self.att1(x2)
        x2p = self.pool(x2)

        x3 = self.cnv21(x2p)
        x3 = self.att2(x3)
        x3p = self.pool(x3)

        x4 = self.cnv31(x3p)
        x4 = self.att3(x4)
        x4p = self.pool(x4)

        x5 = self.cnv41(x4p)
        x5 = self.att4(x5)
        x5p = self.pool(x5)

        x6 = self.cnv51(x5p)
        x6 = self.att5(x6)

        x7 = self.up6(x6)
        x7 = self.fusion1(x5, x7)
        x7 = self.cnv61(torch.cat([x7, x5], dim=1))
        x7 = self.att6(x7)

        x8 = self.up7(x7)
        x8 = self.fusion2(x4, x8)
        x8 = self.cnv71(torch.cat([x8, x4], dim=1))
        x8 = self.att7(x8)

        x9 = self.up8(x8)
        x9 = self.fusion3(x3, x9)
        x9 = self.cnv81(torch.cat([x9, x3], dim=1))
        x9 = self.att8(x9)

        x10 = self.up9(x9)
        x10 = self.fusion4(x2, x10)
        x10 = self.cnv91(torch.cat([x10, x2], dim=1))
        x10 = self.att9(x10)


        if self.last_activation is not None:
            logits = self.last_activation(self.out(x10))

        else:
            logits = self.out(x10)

        return logits
