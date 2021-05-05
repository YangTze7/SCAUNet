import os
import glob
from torch.utils.data import DataLoader, Dataset
from skimage import io
from skimage.transform import resize
import numpy as np
class DFCDataset(Dataset):

    def __init__(self, train_path, train=True,transform=None):



        tmp_list = sorted(glob.glob(os.path.join(train_path,"img*.tif")),key=os.path.getmtime)
        gt_list = sorted(glob.glob(os.path.join(train_path,"gt*.tif")),key=os.path.getmtime)
        self.train = train
        self.mask_list = tmp_list
        self.gt_path = gt_list
        self.transform = transform

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):


        image = io.imread(self.mask_list[idx])
        viirs = image[:,:,94]
        image = np.concatenate((image[:,:,12:24],viirs[:,:,np.newaxis]),axis = 2)
        mask = io.imread(self.gt_path[idx])


       


        if self.transform:

            result_img, mask = self.transform(image, mask)
            img=result_img.transpose(2, 0, 1)

        sample = {"image": img, "label": mask}
        return sample
