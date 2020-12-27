# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 18:33:06 2020

@author: owner
"""
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

root = './dataset/CUB_200_2011/CUB_200_2011'

img_txt_file = open(os.path.join(root, 'images.txt'))
label_txt_file = open(os.path.join(root, 'image_class_labels.txt'))

img_name_list = []

for line in img_txt_file:
    img_name_list.append(line[:-1].split(' ')[-1])
    
 
imgs = [plt.imread(os.path.join(root, 'images', f))
                               for f in img_name_list]

imgs_resize = []
for i in range(len(imgs)):
    temp = imgs[i]
    temp_resize = cv2.resize(temp, dsize= (512, 512), interpolation= cv2.INTER_LINEAR)
    imgs_resize.append(temp_resize)
    
    
    
imgs_rgb = Image.fromarray(imgs_temp, mode='RGB')
img_resize = transforms.Resize((512, 512), Image.BILINEAR)(imgs_rgb)    