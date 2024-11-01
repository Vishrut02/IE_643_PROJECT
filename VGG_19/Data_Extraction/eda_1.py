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

ORIG_DATA_PATH = "marmot_dataset_v1.0/data/English"
POSITIVE_DATA_LBL = os.path.join(ORIG_DATA_PATH, 'Positive', 'Labeled')
DATA_PATH = 'Marmot_data'
PROCESSED_DATA = 'marmot_processed'
IMAGE_PATH = os.path.join(PROCESSED_DATA, 'image')
TABLE_MASK_PATH = os.path.join(PROCESSED_DATA, 'table_mask')
COL_MASK_PATH = os.path.join(PROCESSED_DATA, 'col_mask')

directories = [DATA_PATH, PROCESSED_DATA, IMAGE_PATH, TABLE_MASK_PATH, COL_MASK_PATH]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}" if not os.path.exists(directory) else f"Directory already exists: {directory}")

#utility functions

# Extract Image coordinate from Marmot dataset
# https://stackoverflow.com/a/58521586

def get_table_bbox(table_xml_path,  new_image_shape):
    
    """
    - Extract Table Coordinates from xml
    - Scale them w.r.t to new image shape
    
    Input:
        table_xml_path: str - xml path
        new_image_shape: tuple - (new_h, new_w)
    
    Output:
        table_bboxes: List 
    """

    tree = ET.parse(table_xml_path)
    root = tree.getroot()

    px0, py1, px1, py0 = list(map(lambda x: struct.unpack('!d', bytes.fromhex(x))[0], root.get("CropBox").split()))
    pw = abs(px1 - px0)
    ph = abs(py1 - py0)

    table_bboxes = []

    for table in root.findall(".//Composite[@Label='TableBody']"):
        x0p, y0m, x1p,y1m  = list(map(lambda x: struct.unpack('!d', bytes.fromhex(x))[0], table.get("BBox").split()))
        x0 = round(new_image_shape[1]*(x0p - px0)/pw)
        x1 = round(new_image_shape[1]*(x1p - px0)/pw)
        y0 = round(new_image_shape[0]*(py1 - y0m)/ph)
        y1 = round(new_image_shape[0]*(py1 - y1m)/ph)
        
        table_bboxes.append([x0,y0, x1,y1])
    return table_bboxes


def get_col_bbox(column_xml_path, prev_img_shape, new_image_shape, table_bboxes):
    
    """
    - Extract Column Coordinates from xml
    - Scale them w.r.t to new image shape and prev image shape
    - If there are no table_bboxes present , approximate them using column bbox
    
    Input:
        table_xml_path: str - xml path
        prev_img_shape: tuple - (new_h, new_w)
        new_image_shape: tuple - (new_h, new_w)
        table_bboxes: List - list of table bbox coordinates
    
    Output:
        table_bboxes: List 
    """
    
    tree = ET.parse(column_xml_path)
    root = tree.getroot()
    xmins = [round(int(coord.text) * new_image_shape[1] / prev_img_shape[1]) for coord in root.findall("./object/bndbox/xmin")]
    xmaxs = [round(int(coord.text) * new_image_shape[1] / prev_img_shape[1]) for coord in root.findall("./object/bndbox/xmax")]
    ymins = [round(int(coord.text) * new_image_shape[0] / prev_img_shape[0]) for coord in root.findall("./object/bndbox/ymin")]
    ymaxs = [round(int(coord.text) * new_image_shape[0] / prev_img_shape[0]) for coord in root.findall("./object/bndbox/ymax")]

    col_bbox = []
    for x_min, y_min, x_max, y_max in zip(xmins,ymins,xmaxs,ymaxs):
        bbox = [x_min, y_min, x_max, y_max]
        col_bbox.append(bbox)
    
    #fix 1: if no table coord but have column coord
    if len(table_bboxes) == 0:
        thresh = 3
        x_min = min([x[0] for x in col_bbox]) - thresh 
        y_min = min([x[1] for x in col_bbox]) - thresh 
        x_max = max([x[2] for x in col_bbox]) + thresh  
        y_max = max([x[3] for x in col_bbox]) + thresh 
        
        table_bboxes = [[x_min, y_min, x_max, y_max]]
    
    return col_bbox, table_bboxes

def create_mask(new_h, new_w, bboxes = None):
    
    """
    - create a mask based on new_h, new_w and bounding boxes
    
    Input:
        new_h: int - height of the mask
        new_w: int - width of the mask
        bboxes: List - bounding box coordinates  
    
    Output:
        mask: Image 
    """
    
    mask = np.zeros((new_h, new_w), dtype=np.int32)
    
    if bboxes is None or len(bboxes)==0:
         return Image.fromarray(mask)
    
    for box in bboxes:
        mask[box[1]:box[3], box[0]:box[2]] = 255
    
    return Image.fromarray(mask)

pos_data = glob.glob('marmot_dataset_v1.0/data/English/Positive/Raw' + '/*.bmp')
neg_data = glob.glob('marmot_dataset_v1.0/data/English/Negative/Raw' + '/*.bmp')

fig = plt.figure(figsize = (10, 5))

x = ['Neg Samples', 'Pos Samples']
y = [len(neg_data), len(pos_data)]
plt.bar(x, y,width = 0.4)
plt.title('Distribution of Positive and Negative Samples')
plt.show()

new_h, new_w = 1024, 1024

#Negative example - 1

img_path = 'marmot_dataset_v1.0/data/English/Negative/Raw/10.1.1.1.2000_4.bmp'
image = Image.open(img_path)

#resize image 1024, 1024
image = image.resize((new_h, new_w))

table_mask = create_mask(new_h, new_w)
col_mask = create_mask(new_h, new_w)


f, ax = plt.subplots(1,3, figsize = (20,15))

ax[0].imshow(np.array(image))
ax[0].set_title('Original Image')
ax[1].imshow(table_mask)
ax[1].set_title('Table Mask')
ax[2].imshow(col_mask)
ax[2].set_title('Column Mask')
plt.show()

#Negative example - 2

img_path = 'marmot_dataset_v1.0/data/English/Negative/Raw/10.1.1.1.2016_12.bmp'
image = Image.open(img_path)

#resize imageto std 1024, 1024
image = image.resize((new_h, new_w))

table_mask = create_mask(new_h, new_w)
col_mask = create_mask(new_h, new_w)


f, ax = plt.subplots(1,3, figsize = (20,15))

ax[0].imshow(np.array(image))
ax[0].set_title('Original Image')
ax[1].imshow(table_mask)
ax[1].set_title('Table Mask')
ax[2].imshow(col_mask)
ax[2].set_title('Column Mask')
plt.show()

img_path = 'marmot_dataset_v1.0/data/English/Positive/Raw/10.1.1.1.2006_3.bmp'
table_xml_path = 'marmot_dataset_v1.0/data/English/Positive/Labeled/10.1.1.1.2006_3.xml'
column_xml_path = 'Marmot_data/10.1.1.1.2006_3.xml'

image = Image.open(img_path)

#resize imageto std 1024, 1024
w, h = image.size
image = image.resize((new_h, new_w))

#convert to 3 channel image if 1 channel
if image.mode != 'RGB':
    image = image.convert("RGB")

#scaled versions of bbox coordinates of table
table_bboxes = get_table_bbox(table_xml_path, (new_h, new_w))

#scaled versions of bbox coordinates of columns
col_bboxes, table_bboxes = get_col_bbox(column_xml_path, (h,w), (new_h, new_w), table_bboxes)
col_bboxes

plt.figure(figsize = (20,10))

image_temp = np.array(image).copy()
for bbox in table_bboxes:
    cv2.rectangle(image_temp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

for bbox in col_bboxes:
    cv2.rectangle(image_temp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
    
plt.imshow(image_temp)

table_mask = create_mask(new_h, new_w, table_bboxes)
col_mask = create_mask(new_h, new_w, col_bboxes)

f, ax = plt.subplots(1,3, figsize = (20,15))

ax[0].imshow(np.array(image_temp))
ax[0].set_title('Original Image')
ax[1].imshow(table_mask)
ax[1].set_title('Table Mask')
ax[2].imshow(col_mask)
ax[2].set_title('Column Mask')
plt.show()


img_path = 'marmot_dataset_v1.0/data/English/Positive/Raw/10.1.1.8.2182_6.bmp'
table_xml_path = 'marmot_dataset_v1.0/data/English/Positive/Labeled/10.1.1.8.2182_6.xml'
column_xml_path = 'Marmot_data/10.1.1.8.2182_6.xml'

#load image
image = Image.open(img_path)

#resize imageto std 1024, 1024
w, h = image.size
image = image.resize((new_h, new_w))

#convert to 3 channel image if 1 channel
if image.mode != 'RGB':
    image = image.convert("RGB")

#scaled versions of bbox coordinates of table
table_bboxes = get_table_bbox(table_xml_path, (new_h, new_w))

plt.figure(figsize = (20,10))
plt.imshow(np.array(image))
plt.show()
plt.figure(figsize = (20,10))

image_temp = np.array(image).copy()
for bbox in table_bboxes:
    cv2.rectangle(image_temp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

for bbox in col_bboxes:
    cv2.rectangle(image_temp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
    
plt.imshow(image_temp)

table_mask = create_mask(new_h, new_w, table_bboxes)
col_mask = create_mask(new_h, new_w, col_bboxes)
f, ax = plt.subplots(1,3, figsize = (20,15))

ax[0].imshow(np.array(image_temp))
ax[0].set_title('Original Image')
ax[1].imshow(table_mask)
ax[1].set_title('Table Mask')
ax[2].imshow(col_mask)
ax[2].set_title('Column Mask')
plt.show()

processed_data = []

for i, data in enumerate([neg_data, pos_data]):
    
    for j, img_path in tqdm(enumerate(data)):
        
        image_name = os.path.basename(img_path)
        image = Image.open(img_path)
        w, h = image.size
        
        #convert to RGB image
        image = image.resize((new_h, new_w))
        if image.mode != 'RGB':
            image = image.convert("RGB")
        table_bboxes, col_bboxes = [], []
        
        if i == 1:
            
            #get xml filename
            xml_file = image_name.replace('bmp', 'xml')
            table_xml_path = os.path.join(POSITIVE_DATA_LBL, xml_file)
            column_xml_path = os.path.join(DATA_PATH,xml_file)
            
            #get table boxes
            table_bboxes = get_table_bbox(table_xml_path, (new_h, new_w))
            
            #get column boxes , if table boxes are empty, approximate them using column boxes
            if os.path.exists(column_xml_path):
                col_bboxes, table_bboxes = get_col_bbox(column_xml_path, (h,w), (new_h, new_w), table_bboxes)
            else:
                col_bboxes = []
        
        #generate masks
        table_mask = create_mask(new_h, new_w, table_bboxes)
        col_mask = create_mask(new_h, new_w, col_bboxes)
        
        #save images and masks
        save_image_path = os.path.join(IMAGE_PATH, image_name.replace('bmp', 'jpg'))
        save_table_mask_path = os.path.join(TABLE_MASK_PATH, image_name[:-4] + '_table_mask.png')
        save_col_mask_path = os.path.join(COL_MASK_PATH, image_name[:-4] + '_col_mask.png')
        
        image.save(save_image_path)
        table_mask.save(save_table_mask_path)
        col_mask.save(save_col_mask_path)
        
        #add data to dataframe
        len_table = len(table_bboxes)
        len_cols = len(col_bboxes)

        value = (save_image_path, save_table_mask_path, save_col_mask_path, h, w, int(len_table != 0), \
                 len_table, len_cols, table_bboxes, col_bboxes)
        
        processed_data.append(value)

column_name = ['img_path','table_mask','col_mask','original_height','original_width','hasTable','table_count','col_count','table_bboxes','col_bboxes']
processed_data = pd.DataFrame(processed_data, columns=column_name)

processed_data.to_csv("processed_data.csv", index = False)
processed_data.head()