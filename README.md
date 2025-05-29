# ECHF-Seg

In the field of medical image segmentation, improving model performance with moderate model parameter becomes an urgent and challenge task. To address this problem, a novel model called ECHF-Seg is proposed in this study, which focuses on the effective convolution and high-frequency enhancement scheme. Two corresponding solutions are Adaptive Decomposed Convolution (ADConv) and Dynamic High-frequency Feature enhancement module (DHF), respectively. Specifically, ADConv achieves a balance between the receptive field and computational overhead by establishing a dynamic mapping between the size of the convolution kernel and the number of channels. In the DHF module, a dynamical Discrete Wavelet Transform (DWT) is included for extracting high-frequency component. In particular, the decomposition level of DWT can be adjusted according to the resolution of the input feature maps. The effectiveness of the proposed ECHF-Seg model is validated through five datasets, which are Glas, BUSI, Kvasir, CVC-ClinicDB, and ISIC 2018. Experimental results show that the proposed ECHF-Seg model can greatly outperform other popular models with moderate model parameter. 


## Experiment
In the experimental section, five publicly available and widely utilized datasets are employed for testing purposes. These datasets are:<br> 
ISIC-2018 (dermoscopy, with 2,594 images)<br>
Kvasir-SEG (endoscopy, with 1,000 images)<br> 
BUSI (breast ultrasound, with 647 images)<br> 
CVC-ClinicDB (colonoscopy, with 612 images)<br>  
GlaS (Gland, with 165 images)<br>

In GlaS dataset, we split the dataset into a training set of 85 images and a test set of 80 images. <br>
In ISIC 2018 dataset, we adopt the official split configuration, consisting of a training set with 2,594 images, a validation set with 100 images, and a test set with 1,000 images. <br>
For other dataset, the images are randomly split into training, validation, and test sets with a ratio of 6:2:2.<br>
The dataset path may look like:
```bash
/The Dataset Path/
├── BUSI/
    ├── Train_Folder/
    │   ├── img
    │   ├── labelcol
    │
    ├── Val_Folder/
    │   ├── img
    │   ├── labelcol
    │
    ├── Test_Folder/
        ├── img
        ├── labelcol
```


## Usage

---

### **Installation**
```bash
git clone git@github.com:LDG2333/CFSeg-Net.git
conda create -n cfseg python=3.8
conda activate cfseg
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
``` 


### **Training**
```bash
python train_model.py
```
To run on different setting or different datasets, please modify Config.py .


### **Evaluation**
```bash
python test_model.py
``` 


## Citation

If you find our repo useful for your research, please consider citing our article. <br>
This article has been submitted for peer-review in the journal called *The visual computer*.<br>
```bibtex
@ARTICLE{40046799,
  author  = {Guodong Li, Shiren Li, Yaoxue Lin, Sihua Tang, Wenguang Xu, Kangxian Chen, Guangguang Yang},
  journal = {The Viusal Computer}
  title   = {CFSeg-Net: Context Feature Extraction Network for Medical Image Segmentation},
  year    = {2025}
}
``` 


## Contact

For technical questions, please contact guodong0012@gmail.com .
