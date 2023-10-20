import os
from varname.helpers import debug
import cv2

dataset_dir = 'player_ground_pose_dataset.002'
dataset_classes = os.listdir(dataset_dir)
debug(dataset_classes)
img_with_errors = 0

for dataset_class in dataset_classes:
    error_dataset_class_path = os.path.join(dataset_dir, dataset_class + '_square_error')
    dataset_class_path = os.path.join(dataset_dir, dataset_class)
    for img_name in os.listdir(dataset_class_path):
        if not '.jpg' in img_name:
            continue
        img_path = os.path.join(dataset_class_path, img_name)
        error_img_path = os.path.join(error_dataset_class_path, img_name)
        img = cv2.imread(img_path)
        h, w, c = img.shape
        if h != w:
            img_with_errors += 1
            os.makedirs(error_dataset_class_path, exist_ok=True)
            os.rename(img_path, error_img_path)
            print(f'{img_path}->{error_img_path}')
print(img_with_errors)