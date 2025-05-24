import os
import torch
import ml_collections


save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_cuda = torch.cuda.is_available()
seed = 2
os.environ['PYTHONHASHSEED'] = str(seed)

n_filts = 32
cosineLR = True
n_channels = 3
n_labels = 1
epochs = 1000
img_size = 224
print_frequency = 1
save_frequency = 100
vis_frequency = 100
early_stopping_patience = 100

pretrain = False


task_name = 'Glas'
#task_name = 'BUSI'
#task_name = 'CVC-ClinicDB'
#task_name = 'Kvasir-SEG'
#task_name = 'isic2018'


learning_rate = 1e-3
batch_size = 1

#model_name = 'ACC_UNet'
model_name = 'ECHF'
#model_name = 'ganet'



session_name = 'session3()'  #time.strftime('%m.%d_%Hh%M')
test_session = "session3()"         #

train_dataset = './datasets/'+ task_name+ '/Train_Folder/'
val_dataset = './datasets/'+ task_name+ '/Val_Folder/'
test_dataset = './datasets/'+ task_name+ '/Test_Folder/'
save_path          = task_name +'/'+ model_name +'/' + session_name + '/'
model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'





##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    config.base_channel = 64
    config.n_classes = 1
    return config




