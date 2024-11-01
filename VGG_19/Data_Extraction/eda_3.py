import numpy as np 
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image
import cv2
import os
import xml.dom.minidom

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
import matplotlib.pyplot as plt
import shutil

from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input, Concatenate, UpSampling2D
from keras.models import load_model
import json
import os



source = 'Marmot_data'
destination_1 = 'image'
destination_2 = 'annote'

directories = [destination_1,destination_2]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}" if not os.path.exists(directory) else f"Directory already exists: {directory}")


if os.path.isdir(source):
    data_files = os.listdir(source)
    for file in data_files:
        if (file.endswith("bmp")):
            shutil.move(source+'/'+file,destination_1)
        if (file.endswith("xml")):
            shutil.move(source+'/'+file,destination_2)

data_files1 = os.listdir('Images')
data_files2 = os.listdir('Annotations')
data_files3 = os.listdir('image')
data_files4 = os.listdir('annote')
print(f'Total images from ICDAR 2017 table dataset is {len(data_files1)}')
print("-"*50)
print(f'Total Annotations from ICDAR 2017 table dataset is {len(data_files2)}')
print("-"*50)
print(f'Total images from marmot dataset is {len(data_files3)}')
print("-"*50)
print(f'Total Annotations from marmot dataset is {len(data_files4)}')

image = cv2.imread('image/10.1.1.1.2129_6.bmp',cv2.IMREAD_UNCHANGED)
plt.figure(figsize=(20,6))
plt.imshow(image)

with open('annote/10.1.1.1.2129_6.xml') as xml_file:
    xml_file = xml.dom.minidom.parseString(xml_file.read()) 
    xml_file = xml_file.toprettyxml()
print (xml_file)

#referance :- https://www.geeksforgeeks.org/xml-parsing-python/
dir_path = 'annote'

xmin_ , ymin_ , xmax_ , ymax_ = list(), list(), list() , list()

filename_ , table_width_, table_height_= list(), list(), list() 

for file in os.listdir(dir_path):
  if file.endswith('.xml'):
    tree = ET.parse(dir_path + '/' + file) 
    root = tree.getroot()
    name=root.find("./filename").text
    width=root.find("./size/width").text
    height=root.find("./size/height").text

    xmin , ymin , xmax , ymax = list(), list(), list() , list()


    for x_min in root.findall("./object/bndbox/xmin"):
      xmin.append(x_min.text)
    
    for y_min in root.findall("./object/bndbox/ymin"):
      ymin.append(y_min.text)
    
    for x_max in root.findall("./object/bndbox/xmax"):
      xmax.append(x_max.text)
    
    for y_max in root.findall("./object/bndbox/ymax"):
      ymax.append(y_max.text)

    filename , table_width, table_height = list(), list(), list() 

    for i in range(0,len(xmax)):
      filename.append(name)
      table_width.append(width)
      table_height.append(height)

    for i in range(0,len(xmax)):
      filename_.append(filename[i])
      table_width_.append(table_width[i])
      table_height_.append(table_height[i])

      xmin_.append(xmin[i])
      ymin_.append(ymin[i])
      xmax_.append(xmax[i])
      ymax_.append(ymax[i])


Dict = dict({'filename': filename_, 'table_width': table_width_, 'table_height':table_height_,
           'xmin':xmin_,'ymin':ymin_,'xmax':xmax_,'ymax':ymax_})


df_marmot=pd.DataFrame.from_dict(Dict)
df_marmot.to_pickle('df_marmot.pkl')

##creating table and column mask from marmot dataframe
for m in df_marmot['filename'].unique():
    
    width=int(df_marmot[df_marmot['filename']==m]['table_width'].unique())
    height=int(df_marmot[df_marmot['filename']==m]['table_height'].unique())

    xmin=df_marmot[df_marmot['filename']==m]['xmin'].to_list()
    ymin=df_marmot[df_marmot['filename']==m]['ymin'].to_list()
    xmax=df_marmot[df_marmot['filename']==m]['xmax'].to_list()
    ymax=df_marmot[df_marmot['filename']==m]['ymax'].to_list()


    column_mask = np.zeros((height, width), dtype=np.int32)
    table_mask = np.zeros((height, width), dtype=np.int32)

        
    for k in range(0,len(xmin)):
        xmin[k]=int(xmin[k])
        xmax[k]=int(xmax[k])
        ymin[k]=int(ymin[k])
        ymax[k]=int(ymax[k])

    l1 , l2 = list(),list()
    a = 0
    for j , i  in enumerate(ymin):
      if a == 0:
        a = i
        l2.append(j)
      elif a > 0 and j + 1 < len(ymin):
        if  abs(a-i) <= 50:
          a = i 
          l2.append(j)
        else:
          a = i
          l1.append(l2)
          l2 = []
          l2.append(j)
      else:
        a = i
        l2.append(j)
        l1.append(l2) 

    for k in range(len(l1)):
      x_min = xmin[l1[k][0]:l1[k][len(l1[k])-1]+1]
      y_min = ymin[l1[k][0]:l1[k][len(l1[k])-1]+1]
      x_max = xmax[l1[k][0]:l1[k][len(l1[k])-1]+1]
      y_max = ymax[l1[k][0]:l1[k][len(l1[k])-1]+1]
    
      table_xmin_cordinate=int(min(x_min))
      table_ymin_cordinate=int(min(y_min))
      table_xmax_cordinate=int(max(x_max))
      table_ymax_cordinate=int(max(y_max))
    
      table_mask[table_ymin_cordinate:table_ymax_cordinate, table_xmin_cordinate:table_xmax_cordinate] = 255
    
    for j in range(0,len(xmin)):
        column_mask[int(ymin[j]):int(ymax[j]), int(xmin[j]):int(xmax[j])] = 255
        
    im_col = Image.fromarray(column_mask.astype(np.uint8),'L')
    im_table = Image.fromarray(table_mask.astype(np.uint8),'L')
    im_col.save('marmot_column/'+ m+"_col"+".jpeg")
    im_table.save('marmot_table/' +m+"_table" + ".jpeg")

    #converting marmot image from .bmp to .jpeg file 
file1 = []
for file in os.listdir('marmot_column'):
  if file.endswith('.jpeg'):
    file1.append(file[:-13])


dir_path = 'image'
count = 0

# Iterate over the files in the directory
for filename in os.listdir(dir_path):
    if filename.endswith('.bmp'):
        # Assuming file1 is a predefined list containing filenames (without extension) to check
        if filename[:-4] in file1:
            count += 1
            path = os.path.join(dir_path, filename)
            image = Image.open(path)
            # Resize the image using LANCZOS filter
            image = image.resize((1024, 1024), Image.LANCZOS)
            # Save the resized image as JPEG in the specified directory
            image.save(f'marmots_usuals/{filename[:-4]}.jpeg')

# Set up the figure for visualization
fig = plt.figure(figsize=(15, 15))

  

# reading images
Image1 = mpimg.imread('marmots_usuals/10.1.1.1.2006_3.jpeg')
Image2 = mpimg.imread('marmot_table/10.1.1.1.2006_3.bmp_table.jpeg')
Image3 = mpimg.imread('marmot_column/10.1.1.1.2006_3.bmp_col.jpeg')
  
#ploting marmot image
fig.add_subplot(1, 3, 1)
plt.imshow(Image1)
plt.axis('off')
plt.title("Image")
  
#ploting table mask
fig.add_subplot(1, 3, 2)
plt.imshow(Image2)
plt.axis('off')
plt.title("Table mask")
  
#ploting column mask
fig.add_subplot(1, 3, 3)
plt.imshow(Image3)
plt.axis('off')
plt.title("Column mask")

#storing all the cordinate of each file which capture the tabular format in dataframe
dir_path = 'Annotations'

xmin_ , ymin_ , xmax_ , ymax_ = list(), list(), list() , list()

filename_ , table_width_ ,  table_height_ = list(), list() , list()

for file in os.listdir(dir_path):
  if file.endswith('.xml'):
    name = file[:-4]+".bmp"
    tree = ET.parse('Annotations/' + file)
    root = tree.getroot()

    fname="Images/" + file[:-4]+".bmp"
    img=Image.open(fname)

    xmin , ymin , xmax , ymax = list(), list(), list() , list()
    filename , table_width , table_height = list(), list() , list()
    if(root.findall("tableRegion")):
      data=root.findall("tableRegion/Coords")
      for i in range(0,len(data)):
        point=data[i].get("points")
        filename.append(name)
        table_width.append(img.size[0])
        table_height.append(img.size[1])
        points=point.split(" ")
        points = [[int(num) for num in string.split(",")] for string in points]
        coordinate = []
        for values in points:
          for value in values:
            coordinate.append(value)

        xmin.append(coordinate[0])
        ymin.append(coordinate[1])
        xmax.append(coordinate[6])
        ymax.append(coordinate[7])

      for i in range(0,len(data)):
        filename_.append(filename[i])
        table_width_.append(table_width[i])
        table_height_.append(table_height[i])
        xmin_.append(xmin[i])
        ymin_.append(ymin[i])
        xmax_.append(xmax[i])
        ymax_.append(ymax[i])
    
    else:
      filename_.append(name)
      table_width_.append(img.size[0])
      table_height_.append(img.size[1])
      xmin_.append(0)
      ymin_.append(0)
      xmax_.append(0)
      ymax_.append(0)          

Dict = dict({'filename': filename_,'table_width': table_width_, 'table_height':table_height_,'xmin':xmin_,'ymin':ymin_,'xmax':xmax_,'ymax':ymax_})


df=pd.DataFrame.from_dict(Dict)
df.to_pickle('df_ICDAR.pkl')

for m in df['filename'].unique():
    width=int(df[df['filename']==m]['table_width'].unique())
    height=int(df[df['filename']==m]['table_height'].unique())
    
    xmin=df[df['filename']==m]['xmin'].to_list()
    ymin=df[df['filename']==m]['ymin'].to_list()
    xmax=df[df['filename']==m]['xmax'].to_list()
    ymax=df[df['filename']==m]['ymax'].to_list()

    table_mask = np.zeros((height, width), dtype=np.int32)

    for i in range(len(xmin)):
      table_xmin=xmin[i]
      table_ymin=ymin[i]
      table_xmax=xmax[i]
      table_ymax=ymax[i]
    

      table_mask[table_ymin:table_ymax, table_xmin:table_xmax] = 255 

    im_table = Image.fromarray(table_mask.astype(np.uint8),'L')
    im_table.save('ICDAR_table/' +m+"_table" + ".jpeg")  

    # showing marot image with its corresponding table mask and column mask
fig = plt.figure(figsize=(15, 10))
  

# reading images
Image1 = mpimg.imread('Images/POD_0001.bmp')
Image2 = mpimg.imread('ICDAR_table/POD_0001.bmp_table.jpeg')
  
#ploting marmot image
fig.add_subplot(1, 2, 1)
plt.imshow(Image1)
plt.axis('off')
plt.title("Image")
  
#ploting table mask
fig.add_subplot(1, 2, 2)
plt.imshow(Image2)
plt.axis('off')
plt.title("Table mask")

