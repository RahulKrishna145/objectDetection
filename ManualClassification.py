import os
import cv2
import csv
import json
from pycocotools.coco import COCO

# === Config ===
annotations_path = 'C:\\Users\\aadithya\\Documents\\Armada\\Project\\data\\coco2017\\annotations\\person_keypoints_train2017.json'
images_folder = 'C:\\Users\\aadithya\\Documents\\Armada\\Project\\data\\coco2017\\train2017'
output_csv = 'pose_labels.csv'

# === Init ===
# start_image_id = 000000000009
coco = COCO(annotations_path)
person_cat_id = coco.getCatIds(catNms=['person'])[0]
img_ids = sorted(coco.getImgIds(catIds=[person_cat_id]))
existing_annotations = set()

# === Resume support ===
if os.path.exists(output_csv):
    with open(output_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_annotations.add(int(row['annotation_id']))

# === Output File ===
csv_file = open(output_csv, 'a', newline='')
csv_writer = csv.writer(csv_file)
if os.stat(output_csv).st_size == 0:
    csv_writer.writerow([
        'image_id',
        'file_name',
        'annotation_id',
        'x', 'y', 'width', 'height',
        'category_id',
        'posture',
        'keypoints_json',
        'incomplete_flag'
    ])

print(
    "Press:\n"
    "1 = standing_straight\n"
    "2 = standing_sideways\n"
    "3 = sitting\n"
    "i = mark as INCOMPLETE\n"
    "q = quit"
)

for img_id in img_ids:
    # if img_id < start_image_id:
    #     continue
    img_data = coco.loadImgs(img_id)[0]
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[person_cat_id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    img_path = os.path.join(images_folder, img_data['file_name'])
    if not os.path.exists(img_path):
        continue

    image = cv2.imread(img_path)
    if image is None:
        continue

    for ann in anns:
        if ann['id'] in existing_annotations:
            continue

        x, y, w, h = map(int, ann['bbox'])
        person_crop = image[y:y+h, x:x+w]

        if person_crop.size == 0:
            continue

        display_img = cv2.resize(person_crop, (300, 400))
        cv2.imshow('Person Pose Labeling', display_img)

        key = cv2.waitKey(0)
        incomplete_flag = 'no'

        if key == ord('1'):
            posture = 'standing_straight'
        elif key == ord('2'):
            posture = 'standing_sideways'
        elif key == ord('3'):
            posture = 'sitting'
        elif key == ord('i'):
            posture = 'incomplete'
            incomplete_flag = 'yes'
        elif key == ord('q'):
            print("Exiting.")
            csv_file.close()
            cv2.destroyAllWindows()
            exit()
        else:
            print("Invalid key. Skipping...")
            continue

        keypoints_json = json.dumps(ann['keypoints'])

        csv_writer.writerow([
            img_data['id'],
            img_data['file_name'],
            ann['id'],
            x, y, w, h,
            ann['category_id'],
            posture,
            keypoints_json,
            incomplete_flag
        ])
        csv_file.flush()

csv_file.close()
cv2.destroyAllWindows()
