#import libraries

import os
import math
import cv2
import struct
import glob
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import xml.etree.ElementTree as ET

pos_data = glob.glob('marmot_dataset_v1.0/data/English/Positive/Raw' + '/*.bmp')
neg_data = glob.glob('marmot_dataset_v1.0/data/English/Negative/Raw' + '/*.bmp')

PROCESSED_DATA = 'marmot_processed'
IMAGE_PATH = os.path.join(PROCESSED_DATA, 'image')

new_h, new_w = 1024, 1024
for i, data in enumerate([neg_data, pos_data]):
    
    for j, img_path in enumerate(data):
        
        image_name = os.path.basename(img_path)
        image = Image.open(img_path).convert('LA')
        w, h = image.size
        
        #convert to RGB image
        image = image.resize((new_h, new_w)).convert("RGB")

        #save images and masks
        save_image_path = os.path.join(IMAGE_PATH, image_name.replace('bmp', 'jpg'))
        
        image.save(save_image_path)
processed_data = pd.read_csv('processed_data.csv')
processed_data['img_path'] = processed_data['img_path'].apply(lambda x: x.replace('image', 'image_v2'))
processed_data.to_csv('processed_data_v2.csv', index = False)
processed_data        