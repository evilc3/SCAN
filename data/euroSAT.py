'''
Files to modify : 
1. common_config : add our custom_dataset
2. MyPath : add path to dataset 
3. create  new config files 
4. create new datalaoder


'''


#creating a custom dataloader 
"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import os
import pickle
import sys
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
import pandas as pd



class EuroSat(Dataset):
 
    def __init__(self, root, train=True, transform=None):

        super(EuroSat, self).__init__()

        self.root = root

        
        self.transform = transform
        
        self.train = train  
        
        self.classes = ['Forest','SeaLake','PermanentCrop','Industrial','River','AnnualCrop','HerbaceousVegetation','Residential','Highway','Pasture']

        
        self.df = EuroSat.get_data_dict(self.root)

        #keep 75 for training and 

        if train: 
          self.df = self.df[self.df.train == 1.0].reset_index(drop = True)
        else:
          self.df = self.df[self.df.train == 0.0].reset_index(drop = True)


    @staticmethod
    def get_data_dict(data_dir):

        data_dict = {'path': [], 'label': []}
        
        for (_, labels, _) in os.walk(data_dir):
            for label in labels:
                label_dir = data_dir + label + '/'
                for (_, _, files) in os.walk(label_dir):
                    for file in files:
                        data_dict['path'].append(label_dir + file)
                        data_dict['label'].append(label)

        df =  pd.DataFrame(data_dict)
        #shuffling the dataframe.
        df = df.sample(frac=1).reset_index(drop=True)

        n = len(df)
        #set 75 for training and   and 25 for testing 


        df['train'] = np.zeros(n) 

        df.loc[0:int(n * 0.75),'train'] = 1

        return df


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """  
        img = self.get_image(index)
        img_size = img.size
        
        class_name = self.df['label'][index]
        target = self.classes.index(class_name)


        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': img_size, 'index': index, 'class_name': class_name}}
        
        return out
    
    def get_image(self, index):
        img_path = self.df['path'][index]   
        img = Image.open(img_path)
        return img

    #functions used for plotting images     
    def show_img(self,index):
        img =  self.get_image(index)
        plt.imshow(img)
    
    # will create a batch x batch images 
    def plot_batch(self,batch = 8):
        n_images = len(self.df)

        indices = np.random.choice(range(n_images) , size= batch **2) 

        #creating an image array 
        labels = self.df['label'][indices].reset_index(drop = True)
        images = [np.array(self.get_image(i)) for i in indices]
        
        fig,axis = plt.subplots(batch,batch,figsize = (20,20))


        for i,ax in enumerate(axis.flatten()):

            ax.imshow(images[i])
            ax.set_title(labels[i])
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
             
    
        
    def __len__(self):
        return len(self.df)
