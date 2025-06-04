import os
import csv
from pycocotools.coco import COCO

# === Config ===
annotations_path = 'data\\coco2017\\annotations\\person_keypoints_train2017.json'
images_folder = 'data\\coco2017\\train2017'
output_csv = 'data\\person_bounding_boxes.csv'

# === Init COCO API ===
coco = COCO(annotations_path)
person_cat_id = coco.getCatIds(catNms=['person'])[0]
img_ids = sorted(coco.getImgIds(catIds=[person_cat_id]))

# === Output File ===
with open(output_csv, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([
        'image_id', 'file_name', 'annotation_id',
        'x', 'y', 'width', 'height',
        'area', 'aspect_ratio'
    ])

    for img_id in img_ids:
        img_data = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[person_cat_id])
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            x, y, w, h = ann['bbox']
            area = w * h
            aspect_ratio = round(w / h, 3) if h != 0 else 0

            writer.writerow([
                img_data['id'],
                img_data['file_name'],
                ann['id'],
                int(x), int(y), int(w), int(h),
                int(area),
                aspect_ratio
            ])
