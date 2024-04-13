import os
import pathlib
import torch
from torch.functional import Tensor
from varname.helpers import debug
import cv2
import argparse

def ensure_folder(dir_fname):
    if not os.path.exists(dir_fname):
        try:
            pathlib.Path(dir_fname).mkdir(parents=True, exist_ok=True)
        except PermissionError:
            print('Unable to create {} directory. Permission denied'.format(dir_fname))
            return False
    return os.path.exists(dir_fname)

def img_transform_to_tenzor (input_tenzor) -> Tensor:
    # Imread reads  images as HWC 
    # Torch reads images as CHW
    input_tenzor = input_tenzor.permute(2, 0, 1) # CHW
    input_tenzor = input_tenzor.unsqueeze(0) # 1CHW
    input_tenzor = input_tenzor.to(torch.float32)
    # img_min = input_tenzor.min()
    # img_max = input_tenzor.max()
    # img_tensor = (img_tensor - img_min) / torch.abs(img_max - img_min)
    input_tenzor = input_tenzor  / 255.0
    return input_tenzor

def xywh2x1y1x2y2(xywh_bbox: tuple):
    return (
            xywh_bbox[0],
            xywh_bbox[1],
            xywh_bbox[0] + xywh_bbox[2],
            xywh_bbox[1] + xywh_bbox[3]
        )

def save_dict_to_txt(img_classes, txt_outpath):
    dict_to_txt = open(txt_outpath, 'w')
    for k, v in img_classes.items():
        dict_to_txt.write(str(k) + ','+ str(v) + '\n')
    dict_to_txt.close()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument(
        "--is_debug", default=0, type=int, help="enable fast run on several samples"
    )
    return parser.parse_args()

# TODO перевести в аргпарс и натравить на папку. 
def no_square_img_filter(imgs_dir):
    arguments = parse_arguments()
    dataset_path = arguments.dataset_path
    is_debug = arguments.is_debug
    dataset_classes = os.listdir(dataset_path)
    debug(dataset_classes)
    img_with_errors = 0

    for dataset_class in dataset_classes:
        error_dataset_class_path = os.path.join(dataset_path, dataset_class + '_square_error')
        dataset_class_path = os.path.join(dataset_path, dataset_class)
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
    return img_with_errors