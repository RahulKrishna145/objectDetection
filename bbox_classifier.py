import os
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import cv2

# Ensure pycocotools is installed
try:
    from pycocotools.coco import COCO
except ImportError:
    print("pycocotools not found. Installing...")
    !pip install pycocotools
    from pycocotools.coco import COCO

print("Libraries loaded successfully!")

# Define the base directory where the COCO dataset is located
# IMPORTANT: This path is determined by how you add the dataset to your Kaggle notebook.
# You typically find it under /kaggle/input/ followed by the dataset's short name.
# For 'awsaf49/coco-2017-dataset', it's often something like:
DATA_DIR = '/kaggle/input/coco-2017-dataset/coco2017' # Check your notebook's 'Data' section for the exact path

# Paths for annotations (JSON files)
# Remember, COCO annotations are JSON files, not a single file for the whole dataset.
ANNOTATION_FILE_TRAIN = os.path.join(DATA_DIR, 'annotations', 'person_keypoints_train2017.json')
ANNOTATION_FILE_VAL = os.path.join(DATA_DIR, 'annotations', 'person_keypoints_val2017.json')

# Paths for images
IMAGE_DIR_TRAIN = os.path.join(DATA_DIR, 'train2017') # Note: Some Kaggle versions might have another 'train2017' subdirectory
IMAGE_DIR_VAL = os.path.join(DATA_DIR, 'val2017')     # Same here

# You might need to adjust IMAGE_DIR_TRAIN/VAL if the structure is like:
# /kaggle/input/coco-2017-dataset/train2017/train2017/...
# In that case, it would be:
# IMAGE_DIR_TRAIN = os.path.join(DATA_DIR, 'train2017', 'train2017')
# IMAGE_DIR_VAL = os.path.join(DATA_DIR, 'val2017', 'val2017')

print(f"Checking paths...")
print(f"Annotation file (train): {ANNOTATION_FILE_TRAIN} - Exists: {os.path.exists(ANNOTATION_FILE_TRAIN)}")
print(f"Annotation file (val): {ANNOTATION_FILE_VAL} - Exists: {os.path.exists(ANNOTATION_FILE_VAL)}")
print(f"Image directory (train): {IMAGE_DIR_TRAIN} - Exists: {os.path.exists(IMAGE_DIR_TRAIN)}")
print(f"Image directory (val): {IMAGE_DIR_VAL} - Exists: {os.path.exists(IMAGE_DIR_VAL)}")

# Initialize COCO API for validation annotations
# This is the correct way to load COCO annotations
try:
    coco_val = COCO(ANNOTATION_FILE_VAL)
    print("\nCOCO API initialized successfully for validation set!")
except Exception as e:
    print(f"Error initializing COCO API: {e}")
    print("Please double-check your ANNOTATION_FILE_VAL path.")

# From here, you acan proceed with the steps we discussed previously:
# - Get categories
# - Get image IDs
# - Load image info
# - Get annotations for an image
# - Visualize annotations

people_set = []
if 'coco_val' in locals(): # Proceed only if COCO API was initialized successfully
    img_ids = coco_val.getImgIds()
    
    if img_ids:
        for image in img_ids:
            img_id = image
            img_info = coco_val.loadImgs(img_id)[0]
            #print(f"\n selected image file name: {img_info['file_name']}")
    
            ann_ids = coco_val.getAnnIds(imgIds=img_info['id'])
            annotations = coco_val.loadAnns(ann_ids)
            #cats = coco_val.getCatIds()
            #categories = coco_val.loadCats(cats)
            #cat_map = {cat['id']:cat['name'] for cat in categories}
            people_anns = coco_val.getAnnIds(imgIds = img_info['id'],catIds = 1)
            people_in_img = coco_val.loadAnns(people_anns)
            person = {}
            for people in people_in_img:
                size = ""
                #size determination
                x,y,w,h = people["bbox"]
                if h<100 or w<40:
                    size = "small"
                elif (h>=100 and h<200) and (w>=40 and w<80):
                    size = "medium"
                elif h>=200 and w>=80:
                    size = "large"
                person = {"id":people['id'],"image_id":people["image_id"],"bbox":people['bbox'],"size":size}
                people_set.append(person)
            #for items in annotations:
            #    print("category_id:",items['category_id']," name:",cat_map[items['category_id']])
            #print(f"Number of objects in this image: {len(annotations)}")
    
            # Load and display the image (example)
    
            '''
            image_path = os.path.join(IMAGE_DIR_VAL, img_info['file_name'])
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                plt.figure(figsize=(10, 8))
                ax = plt.gca()
                plt.imshow(image)
                plt.axis('off')
                coco_val.showAnns(annotations, draw_bbox=True)
                #for anns in annotations:
                #    x,y,w,h = anns['bbox']
                #    cat_name = cat_map.get(anns['category_id'])
    
                    #rect = patches.Rectangle((x,y),w,h,linewidth = 2,edgecolor = 'red',facecolor = 'none')
                    #ax.add_patch(rect)
                #    plt.text(x,y-5,cat_name,color = 'black',fontsize = 9)
                    
                plt.title(f"Image ID: {img_info['id']}")
                plt.show()
            else:
                print(f"Could not load image: {image_path}")
            '''
    #print(people_set)
    people_df = pd.DataFrame(people_set)
    print(people_df.head())
