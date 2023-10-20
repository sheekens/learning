import os
import numpy as np
import cv2
from varname.helpers import debug
from sports_dataloader_count_finder import load_img_paths

keyboard_classes_mapping = {
    110: 'moving',
    109: 'static'
    }

def markup(dataset_path, outpath):
    programm_finish = False
    img_paths = load_img_paths(dataset_path)
    img_classes = dict.fromkeys(img_paths.keys(), None)
    cur_img_id = min(img_paths.keys())
    while programm_finish == False:
        cur_img_path = img_paths[cur_img_id]
        cur_img = cv2.imread(cur_img_path)
        cur_img = cv2.putText(
            cur_img,
            str(img_classes[cur_img_id]),
            (20, 20),
            1,
            2,
            (200, 100, 240),
            2
        )
        cv2.imshow('test', cur_img)
        pressed_key = cv2.waitKey(0)
        if pressed_key == 44: #, na angl raskladke
            cur_img_id -= 1
        if pressed_key == 46: #. na angl raskladke
            cur_img_id += 1
        if pressed_key == 122: #z na angl raskladke
            programm_finish = True
        if pressed_key == 110: #n na angl raskladke
            img_classes[cur_img_id] = keyboard_classes_mapping[110]
        if pressed_key == 109: #m na angl raskladke
            img_classes[cur_img_id] = keyboard_classes_mapping[109]
        print(pressed_key)
        if cur_img_id not in img_paths.keys():
            if cur_img_id > max(img_paths.keys()):
                cur_img_id = min(img_paths.keys())
            if cur_img_id < min(img_paths.keys()):
                cur_img_id = max(img_paths.keys())
        
    # for frame, path in img_paths.items():
        # cur_class = img_classes[frame]
        # print(img_classes)

    # return img_paths


cur_path = 'learning/datasets/sportsMOT_volley_starter_pack/sportsMOT_volley_light_dataset/img1/000075.jpg'
dataset_path = 'datasets/sportsMOT_volley_starter_pack/sportsMOT_volley_light_dataset'
markup(dataset_path, None)