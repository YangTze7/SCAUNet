import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
#
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda.amp import autocast
from utils import train_net
from dataset.DFCDataset import DFCDataset
from SEUNet import Unet

Image.MAX_IMAGE_PIXELS = 1000000000000000

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda")


import glob

import numpy as np
from torchsat.transforms import transforms_det

train_transform = transforms_seg.Compose([
    transforms_seg.RandomVerticalFlip(p=0.5),
    transforms_seg.RandomHorizontalFlip(p=0.5),
    # transforms_seg.RandomShift(max_percent=0.4),
    # transforms_seg.RandomRotationY(),
])

imgs_dirs = '/content/train_data'

val_ratio = 0.2
random_state = 42

mass_dataset = DFCDataset(train_path = imgs_dirs, transform=train_transform)
sample_nums = len(mass_dataset)
sample_nums_train = sample_nums*(1-val_ratio)
train_data, valid_data = torch.utils.data.random_split(mass_dataset, [int(sample_nums_train), sample_nums-int(sample_nums_train)])



model = Unet(12,2).cuda()

model_name = "SEUNet"
save_ckpt_dir = os.path.join('/checkpoints/seunet_i12_aug/', model_name, 'ckpt')
save_log_dir = os.path.join('/checkpoints/seunet_i12_aug/', model_name)
if not os.path.exists(save_ckpt_dir):
    os.makedirs(save_ckpt_dir)
if not os.path.exists(save_log_dir):
    os.makedirs(save_log_dir)


param = {}

param['epochs'] = 300         
param['batch_size'] = 16     
param['lr'] = 1e-2            
param['gamma'] = 0.2          
param['step_size'] = 5        
param['momentum'] = 0.9       
param['weight_decay'] = 5e-4    
param['disp_inter'] = 1       
param['save_inter'] = 4       
param['iter_inter'] = 50     
param['min_inter'] = 10

param['model_name'] = model_name          
param['save_log_dir'] = save_log_dir      
param['save_ckpt_dir'] = save_ckpt_dir    


param['load_ckpt_dir'] = None


# if __name__ == '__main__':
best_model, model = train_net(param, model, train_data,valid_data,plot=True)

