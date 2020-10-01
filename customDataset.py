from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class CustomDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        '''
        Args:
            csv_file (string): Path to the csv file with annotations/list of images or videos.
            root_dir (string): Directory with all the images/videos.
            transform (callable, optional): Optional transform to be applied on each sample.
        '''
        
        self.all_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        # Returns length of the dataset, by computing length of annotations/samples
        return len(self.all_data)


    def __getitem__(self, idx):
        '''
        Returns a sample with index [idx] in the annotation list.
        Applies all preprocessing and transformation as needed.
        '''
        
        # Convert idx to list if not (if call made by torch dataloader)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Path to actual data
        # Assumes all_data has M rows of samples, 
        #   with each sample having N columns of annotation (name, label, etc)
        # This assumes first column is name of the sample (filename)
        img_name = os.path.join(self.root_dir,
                                self.all_data.iloc[idx, 0])
        
        # Read the actual image. If video, read video as numpy array and pick frames.
        image = io.imread(img_name)
        '''
        # Alternative image readers
        image = PIL.Image.open(img_name)
        image = cv2.imread(img_name)
        
        # Video reader 
        from skvideo.io import vread
        video = vread(vid_name)     # Returns video as array [F, H, W, C]
        
        # OpenCV video reader
        video = cv2.VideoCapture(vid_name)  # This opens video file handle, have to manually read all the frames
        '''
        
        # Get the lable and typecast to correct data type
        label = self.all_data.iloc[idx, 1]
        label = int(label)
        
        # Other optional annotations
        bbox = self.all_data.iloc[idx, 2:6]     # bbox = [5,3,15,22] -> list of int/string 
        bbox = np.array(bbox).astype('float').reshape(-1, 4)    
        #           bbox = [[5.0, 3.0, 15.0, 22.0]] -> numpy float array, shape 1x4
        
        # Apply transformations if given (crop, flip, resize, etc)
        if self.transform:
            image = self.transform(image)
        
        sample = [image, label]
        # sample = [image, label, bbox] 
        
        return sample