import os
import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import cv2


bbox_height = 0
import numpy as np

def angle(a, b, c):
    # Convert points to NumPy arrays
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Vectors BA and BC
    ba = a - b
    bc = c - b
    
    # Compute the cosine of the angle using dot product formula
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    
    # Clip cosine_angle to the valid range [-1, 1] to avoid numerical errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Compute the angle in radians and then convert to degrees
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def body_angles(point_map):
    leftknee_ang = None
    rightknee_ang = None
    lefthip_ang = None
    righthip_ang = None
    if point_map['left_knee'] != None and point_map['left_ankle'] !=None and point_map['left_hip'] != None:
        leftknee_ang = angle(point_map['left_hip'][0:2],point_map['left_knee'][0:2],point_map['left_ankle'][0:2])
        
    
    print("left knee angle is: -----",leftknee_ang,"-----")
    if point_map['right_knee'] != None and point_map['right_ankle'] !=None and point_map['right_hip'] != None:
        rightknee_ang = angle(point_map['right_hip'][0:2],point_map['right_knee'][0:2],point_map['right_ankle'][0:2])
        
    
    print("right knee angle is: -----",rightknee_ang,"-----")

    
    if point_map['left_shoulder'] != None and point_map['left_hip'] !=None and point_map['left_knee'] != None:
        lefthip_ang = angle(point_map['left_shoulder'][0:2],point_map['left_hip'][0:2],point_map['left_knee'][0:2])
        
    
    print("left hip angle is: -----",lefthip_ang,"-----")
    
    if point_map['right_shoulder'] != None and point_map['right_knee'] !=None and point_map['right_hip'] != None:
        righthip_ang = angle(point_map['right_shoulder'][0:2],point_map['right_hip'][0:2],point_map['right_knee'][0:2])
    
    print("right hip angle is: -----",righthip_ang,"-----")

    return {"left_knee":leftknee_ang,"right_knee":rightknee_ang,"left_hip":lefthip_ang,"right_hip":righthip_ang}

def get_pose(joint_angles,point_map,bbox_height):
    pose = None
    vertical_tol_small = 0.05*bbox_height
    vertical_tol_medium = 0.15*bbox_height
    angle_knee_bent_threshold = 120
    angle_knee_straight_threshold = 150
    angle_hip_bent_threshold = 120
    angle_hip_straight_threshold = 150
    avg_hip_y = 0
    avg_shoulder_y = 0
    #average hip height
    if point_map['left_hip'] and point_map['right_hip']:
        avg_hip_y = (point_map['left_hip'][1] + point_map['right_hip'][1])/2
    elif point_map['left_hip']:
        avg_hip_y = point_map['left_hip'][1]
    elif point_map['right_hip']:
        avg_hip_y = point_map['right_hip'][1]

    #average shoulder height
    if point_map['left_shoulder'] and point_map['right_shoulder']:
        avg_shoulder_y = (point_map['left_shoulder'][1] + point_map['right_shoulder'][1])/2
    elif point_map['left_shoulder']:
        avg_shoulder_y = point_map['left_shoulder'][1]
    elif point_map['right_shoulder']:
        avg_shoulder_y = point_map['right_shoulder'][1]
    
    y_torso = abs(avg_shoulder_y - avg_hip_y)
    
    

# Condition 1: hips are significantly lower than shoulders
    s1 = avg_hip_y > avg_shoulder_y + vertical_tol_medium
    
    # Condition 2: at least one knee or hip is bent
    s2 = (
    ((joint_angles.get('left_knee') is not None and joint_angles['left_knee'] < angle_knee_bent_threshold) and
     (joint_angles.get('right_knee') is not None and joint_angles['right_knee'] < angle_knee_bent_threshold))
    or
    ((joint_angles.get('left_hip') is not None and joint_angles['left_hip'] < angle_hip_bent_threshold) and
     (joint_angles.get('right_hip') is not None and joint_angles['right_hip'] < angle_hip_bent_threshold))
    )

    # Condition 3: vertical distance between hip and knee is relatively small
    s3 = any([
        point_map.get('left_hip') and point_map.get('left_knee') and \
            abs(point_map['left_hip'][1] - point_map['left_knee'][1]) < y_torso * 0.6,
        point_map.get('right_hip') and point_map.get('right_knee') and \
            abs(point_map['right_hip'][1] - point_map['right_knee'][1]) < y_torso * 0.6
    ])
    # Final decision
    if (s1 and s2) or (s1 and s3):
        pose = "sitting"

        # --- Standing logic ---

    # Condition 1: hips are above or at the level of shoulders (upright posture)
    stand1 = avg_hip_y > avg_shoulder_y + vertical_tol_small

    # Condition 2: both knees and hips are relatively straight
    # Count number of joints that are considered "straight"
    straight_joint_count = 0

    if joint_angles.get('left_knee') is not None and joint_angles['left_knee'] > angle_knee_straight_threshold:
        straight_joint_count += 1
    if joint_angles.get('right_knee') is not None and joint_angles['right_knee'] > angle_knee_straight_threshold:
        straight_joint_count += 1
    if joint_angles.get('left_hip') is not None and joint_angles['left_hip'] > angle_hip_straight_threshold:
        straight_joint_count += 1
    if joint_angles.get('right_hip') is not None and joint_angles['right_hip'] > angle_hip_straight_threshold:
        straight_joint_count += 1

    # Standing if at least 3 of the 4 joints are straight
    stand2 = straight_joint_count >= 2   

    
    

    # Condition 3: vertical leg extension (ankle to hip height) is long enough
    stand3 = any([
        point_map.get('left_ankle') and point_map.get('left_hip') and \
            abs(point_map['left_ankle'][1] - point_map['left_hip'][1]) / bbox_height > 0.45,
        point_map.get('right_ankle') and point_map.get('right_hip') and \
            abs(point_map['right_ankle'][1] - point_map['right_hip'][1]) / bbox_height > 0.45
    ])

    # Final standing decision
    if (stand1 and stand2) or (stand1 and stand3):
        pose = "standing"

    
    return pose

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
DATA_DIR = '/kaggle/input/coco-subset-for-pose-estimation' # Check your notebook's 'Data' section for the exact path

# Paths for annotations (JSON files)
# Remember, COCO annotations are JSON files, not a single file for the whole dataset.
ANNOTATION_FILE_TRAIN = os.path.join(DATA_DIR,'dataset','annotations_trainval2017', 'person_keypoints_train2017.json')
ANNOTATION_FILE_VAL = os.path.join(DATA_DIR,'dataset','annotations_trainval2017', 'person_keypoints_val2017.json')
# Paths for images
IMAGE_DIR_TRAIN = os.path.join(DATA_DIR,'dataset','train') # Note: Some Kaggle versions might have another 'train2017' subdirectory
IMAGE_DIR_VAL = os.path.join(DATA_DIR,'dataset','val')     # Same here

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
person_found = 0
# Example of getting image and annotation info:
if 'coco_val' in locals(): # Proceed only if COCO API was initialized successfully
    img_ids = coco_val.getImgIds()
    if img_ids:
        while not person_found:
            random_img_id = random.choice(img_ids)
            img_info = coco_val.loadImgs(random_img_id)[0]
            print(f"\nRandomly selected image file name: {img_info['file_name']}")
    
            ann_ids = coco_val.getAnnIds(imgIds=img_info['id'])
            annotations = coco_val.loadAnns(ann_ids)
            cats = coco_val.getCatIds()
            categories = coco_val.loadCats(cats)
            cat_map = {cat['id']:cat['name'] for cat in categories}
            #kp_list = categories[0]['keypoints']
            #skeleton = categories[0]['skeleton']
            #print(annotations)
            
            for anns in annotations:
                coordinate_array = [] #list of coordinate tuple (x,y,v)
                point_map = {}
                if anns['category_id'] == 1:
                    person_found = 1
                    keypoints = anns['keypoints']
                    for i in range(0,51,3):
                        if keypoints[i+2] > 0:
                            point=(keypoints[i],keypoints[i+1],keypoints[i+2])#forming coordinate array
                        else:
                            point = None
                        coordinate_array.append(point)
                    point_map = dict(zip(kp_list,coordinate_array))#mapping part to coordinates 
                    print(point_map)
                    #angle calculation
                    joint_angles = body_angles(point_map)
                    bbox_height = anns['bbox'][3]
                    pose = get_pose(joint_angles,point_map,bbox_height)
                    print(pose)
                    #print(anns)
                    
                    image_path = os.path.join(IMAGE_DIR_VAL, img_info['file_name'])
                    if os.path.exists(image_path):
                        image = Image.open(image_path).convert('RGB')
                        plt.figure(figsize=(10, 8))
                        ax = plt.gca()
                        plt.imshow(image)
                        plt.axis('off')
                        coco_val.showAnns(annotations, draw_bbox=True)
                        x,y,w,h = anns['bbox']
                        cat_name = cat_map.get(anns['category_id'])
                        
                        rect = patches.Rectangle((x,y),w,h,linewidth = 2,edgecolor = 'red',facecolor = 'none')
                        ax.add_patch(rect)
                        plt.text(x,y-5,cat_name,color = 'black',fontsize = 9)
                            
                        plt.title(f"Image ID: {img_info['id']}")
                        plt.show()
                    else:
                        print(f"Could not load image: {image_path}")
                else:
                    continue
