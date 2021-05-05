import glob
import os

import gdal
import numpy as np
from skimage import io
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

def read_img(filename):
    dataset=gdal.Open(filename) 

    im_width = dataset.RasterXSize  
    im_height = dataset.RasterYSize  
    im_bands = dataset.RasterCount   

    im_geotrans = dataset.GetGeoTransform()  
    im_proj = dataset.GetProjection() 
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)

    del dataset 

    return im_data



def write_img(im_data,filename):

    #gdal.GDT_Byte, 
    #gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    #gdal.GDT_Float32, gdal.GDT_Float64


    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32


    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape 


    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)



    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])

    del dataset

train_path = "/Train"
for parent,tile,_ in os.walk(train_path):
    tiles = [os.path.join(train_path,p) for p in tile]
    break

tmp_list = []
gt_list = []
for tile in tqdm(tiles):

    sar_path = [os.path.join(tile,p) for p in glob.glob(os.path.join(tile,"S1A*.tif"))]
    # sar_path = sar_path[4:6]


    s2_path = [os.path.join(tile,p) for p in glob.glob(os.path.join(tile,"L2A*.tif"))]


    lc8_path = [os.path.join(tile,p) for p in glob.glob(os.path.join(tile,"LC08*.tif"))]


    viirs_path = [os.path.join(tile,p) for p in glob.glob(os.path.join(tile,"DNB*.tif"))]

    tmp_list.append(s2_path + lc8_path+sar_path+viirs_path)
    gt_list.append(os.path.join(tile,"groundTruth.tif"))

save_ckpt_dir = "/train_data"
if not os.path.exists(save_ckpt_dir):
    os.makedirs(save_ckpt_dir)

idx = 0
for maskpath ,gt_path in tqdm(zip(tmp_list,gt_list)):

    # maskpath = self.mask_list[idx]
    image = [read_img(i) for i in maskpath]
    image = np.array(image,'float')
    image[0:80] = image[0:80]/10000.0
    image[89:98] = image[89:98]/10.0


    label = io.imread(gt_path)

    label[label>1] = 0
    label = resize(label,(800,800),order=0,mode='edge',preserve_range=True)
    label = np.array(label,'uint8')
    # 800*800->256*256 stride = 128



    for i in range(6):
        for j in range(6):
            x_patch = image[:,i*128:(i+2)*128,j*128:(j+2)*128]
            y_patch = label[i*128:(i+2)*128,j*128:(j+2)*128]

            if(j==5 and i!=5):
                x_patch = image[:,i*128:(i+2)*128,544:800]
                y_patch = label[i*128:(i+2)*128,544:800]

            if(j!=5 and i==5):
                x_patch = image[:,544:800,j*128:(j+2)*128]
                y_patch = label[544:800,j*128:(j+2)*128]
            
            if(j==5 and i==5):
                x_patch = image[:,544:800,544:800]
                y_patch = label[544:800,544:800]
            
            write_img(x_patch,os.path.join(save_ckpt_dir,"img_"+str(idx).zfill(4)+".tif"))
            write_img(y_patch,os.path.join(save_ckpt_dir,"gt_"+str(idx).zfill(4)+".tif"))
            idx+=1

print("end")

