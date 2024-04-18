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

def square_from_rectangle(player_bbox: tuple):
    x, y, w, h = player_bbox
    centr = [int(x+w/2), int(y+h/2)]
    square_side = max(w, h)
    x1y1x2y2 = [
        int(centr[0] - square_side/2), 
        int(centr[1] - square_side/2),
        int(centr[0] + square_side/2), 
        int(centr[1] + square_side/2)             
    ]
    x1y1x2y2_to_return = []
    for value in x1y1x2y2:
        if value < 0:
            value = 0
        x1y1x2y2_to_return.append(value)
    return x1y1x2y2_to_return

def save_dict_to_txt(img_classes, txt_outpath):
    dict_to_txt = open(txt_outpath, 'w')
    for k, v in img_classes.items():
        dict_to_txt.write(str(k) + ','+ str(v) + '\n')
    dict_to_txt.close()

def load_classes_from_txt(txt_outpath): 
    img_classes_txt= None
    img_classes = {}
    with open(txt_outpath, 'r') as f:
        img_classes_txt = f.readlines()
    for img_object in img_classes_txt: 
        img_object = img_object.replace('\n', '').split(',')
        for i in range(len(img_object)):
            img_object[i] = img_object[i]
        cur_img_id, img_class = img_object
        cur_img_id = int(cur_img_id)
        if cur_img_id not in img_classes.keys(): 
            img_classes[cur_img_id] = {}
        img_classes[cur_img_id] = img_class
    return img_classes

# usage: py .\file_name.py --dataset_path D:\~ --is_debug True
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument(
        "--is_debug", default=0, type=int, help="enable fast run on several samples"
    )
    return parser.parse_args()

def no_square_img_filter_by_classes(dataset_path):
    arguments = parse_arguments()
    dataset_path = arguments.dataset_path
    is_debug = arguments.is_debug
    dataset_classes = os.listdir(dataset_path)
    img_with_errors = 0

    for dataset_class in dataset_classes:
        error_dataset_class_path = os.path.join(dataset_path, dataset_class + '_square_error')
        dataset_class_path = os.path.join(dataset_path, dataset_class)
        for img_name in os.listdir(dataset_class_path):
            if not '.png' in img_name:
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

def no_square_img_filter(dataset_path):
    arguments = parse_arguments()
    dataset_path = arguments.dataset_path
    is_debug = arguments.is_debug
    img_with_errors = []

    for img_name in os.listdir(dataset_path):
        if not '.jpg' in img_name:
            continue
        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path)
        h, w, c = img.shape
        if h != w:
            img_with_errors.append(img_name)
    return img_with_errors

def run_dataset(dataset_path, outdir, output_save=False): 
    img_paths = load_img_paths(dataset_path)
    gt_objects = load_gt(dataset_path)
    colors_list = [
        (255,0,0),
        (0,255,0),
        (0,0,255),
        (255,255,0),
        (255,0,255),
        (0,255,255),
        (100,0,0),
        (0,100,0),
        (0,0,100),
        (100,100,0),
        (100,0,100),
        (255,0,100)
    ]
    for frame_number, frame_objects in gt_objects.items():
        try: 
            img_path = img_paths[frame_number]
        except KeyError:
            continue
        img = cv2.imread(img_path)
        img2draw = img.copy() 
        for player_id, player_bbox in frame_objects.items():
            cv_bbox = xywh2x1y1x2y2(player_bbox)
            cv2.rectangle(
                    img2draw, 
                    #top left 
                    pt1=(cv_bbox[0], cv_bbox[1]),
                    #bottom rigth
                    pt2=(cv_bbox[2], cv_bbox[3]),
                    color=colors_list[player_id],
                    thickness=2)
            img2draw = cv2.putText(
                img2draw,
                str(player_id),
                (cv_bbox[0], cv_bbox[1]-3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                colors_list[player_id],
                2
                )
            img2draw = cv2.circle(
                img2draw,
                (
                (xywh2x1y1x2y2(player_bbox)[0]+int((xywh2x1y1x2y2(player_bbox)[2]-xywh2x1y1x2y2(player_bbox)[0])/2)),
                (xywh2x1y1x2y2(player_bbox)[1]+int((xywh2x1y1x2y2(player_bbox)[3]-xywh2x1y1x2y2(player_bbox)[1])/2))
                ),
                2,
                colors_list[player_id],
                -1
            )
            # print(int((xywh2x1y1x2y2(player_bbox)[2]-xywh2x1y1x2y2(player_bbox)[0])/2))

        if output_save:
            os.makedirs(outdir, exist_ok=True)
            img_out_path = os.path.join(outdir, os.path.basename(img_path))
            cv2.imwrite(img_out_path, img2draw)
            print('out written to', os.path.abspath(img_out_path))

def load_gt(dataset_path: str): 
    gt_path = None
    gt_data= None
    gt_objects = {}
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if 'gt.txt' == file_name: 
                gt_path = os.path.join(root, file_name)
    if not gt_path: 
        return None
    with open(gt_path, 'r') as f:
        gt_data = f.readlines()
    for gt_object in gt_data: 
        gt_object = gt_object.replace('\n', '').split(', ')
        for i in range(len(gt_object)):
            gt_object[i] = int(gt_object[i])
        frame_number, player_id, top_left_x, top_left_y, width, height, _, _, _ = gt_object 
        if frame_number not in gt_objects.keys(): 
            gt_objects[frame_number] = {}
        gt_objects[frame_number][player_id] = (top_left_x, top_left_y, width, height)
    return gt_objects

def load_img_paths(dataset_path): 
    img_paths = {}
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if '.jpg' in file_name and len(file_name) == 10: 
                frame_number = int(file_name[:-4])
                img_paths[frame_number] = os.path.join(root, file_name)
    return img_paths