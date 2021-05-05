import ttach as tta
import os
from skimage import io
import glob
import torch

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageOps
from SEUNet import Unet

model = Unet(12,2).cuda()



def ds1(arr):
    arr1 = np.zeros((16,16))
    for i in range(16):
        for j in range(16):
            patch =  arr[i*50:(i+1)*50,j*50:(j+1)*50]
            if np.mean(patch)>0:
                arr1[i,j] = 1
    return arr1


def test(val_path,result_path):

    os.makedirs(result_path,exist_ok=True)
    tmp_list = []
    tiles = []
    for parent,tile,_ in os.walk(val_path):
        tiles = [os.path.join(val_path,p) for p in tile]
        break

    for tile in tqdm(tiles):

        sar_path = [os.path.join(tile,p) for p in glob.glob(os.path.join(tile,"S1A*.tif"))]


        s2_path = [os.path.join(tile,p) for p in glob.glob(os.path.join(tile,"L2A*.tif"))]


        lc8_path = [os.path.join(tile,p) for p in glob.glob(os.path.join(tile,"LC08*.tif"))]


        viirs_path = [os.path.join(tile,p) for p in glob.glob(os.path.join(tile,"DNB*.tif"))]

        tmp_list.append(s2_path + lc8_path+sar_path+viirs_path)



    checkpoint = torch.load("seunet12-checkpoint-best.pth")
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')



    for maskpath in tmp_list:

        image = [io.imread(i) for i in maskpath]
        image = np.array(image,'float')
        image[0:80] = image[0:80]/10000.0
        image[89:98] = image[89:98]/10.0
        image=image.transpose(1, 2, 0)
        viirs = image[:,:,94]
        image = np.concatenate((image[:,:,12:24],viirs[:,:,np.newaxis]),axis = 2)
        
        image_padding = np.zeros((1152,1152,13))
        for i in range(13):
            band = image[:,:,i]
            band = np.pad(band,((0,224),(0,224)),'edge')

            band = np.pad(band,((64,64),(64,64)),'edge')
            image_padding[:,:,i] = band
        


        
        image = image_padding.transpose(2, 0, 1)



        labels = np.zeros((64,256,256))
        label = np.zeros((1024,1024))
        label_final = np.zeros((800,800))
        # label = 

        for i in range(8):
            for j in range(8):
                x_patch = image[:,i*128:(i+2)*128,j*128:(j+2)*128]


                inputs = torch.from_numpy(x_patch).unsqueeze(0).float()
                inputs = inputs.cuda()
                v = inputs[:,12,:,:]
                s2 = inputs[:,0:12,:,:]
                # pred = model(v[:,np.newaxis,:,:],s2)
                with torch.no_grad():
                    output = tta_model(v[:,np.newaxis,:,:],s2)

                pred = output.squeeze().cpu().data.numpy()
                pred = np.argmax(pred,axis=0)


                pred_map = pred.astype("uint8")
                labels[i*8+j,:,:] = pred_map


        for i in range(8):
            for j in range(8):


                label[i*128:(i+1)*128,j*128:(j+1)*128] = labels[i*8+j,64:192,64:192]
        
        label_final = label[0:800,0:800]
        label_final1 = ds1(label_final)

        im = Image.fromarray(label_final1)
        path1,_ = os.path.split(maskpath[0])
        _,path2 = os.path.split(path1)

        save_file_name = path2[4:]+".tif"
        
        save_path = os.path.join(result_path,save_file_name)

        im.save(save_path)
        print(save_path)

test('/Test/','/result1')
