import os
import numpy as np
import cv2
from varname.helpers import debug
from sports_dataloader_count_finder import load_img_paths

keyboard_classes_mapping = {
    110: 'moving',
    109: 'static'
    }

def load_classes_form_txt(txt_outpath): 
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

def markup(dataset_path, txt_outpath, load_previous_markup, write_classes_to_txt):
    programm_finish = False
    img_paths = load_img_paths(dataset_path)
    if load_previous_markup == False:
        img_classes = dict.fromkeys(img_paths.keys(), None)
    else:
        img_classes = load_classes_form_txt(txt_outpath)
    cur_img_id = min(img_paths.keys())
    while programm_finish == False:
        cur_img_path = img_paths[cur_img_id]
        cur_img = cv2.imread(cur_img_path)
        cur_img = cv2.putText(
            cur_img,
            str('frame {} class {}'.format(cur_img_id, img_classes[cur_img_id])),
            (10, 25),
            1,
            1.7,
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
    if write_classes_to_txt == True:
        save_to_txt(img_classes, txt_outpath)
    return img_classes

def save_to_txt(img_classes, txt_outpath):
    dict_to_txt = open(txt_outpath, 'w')
    for k, v in img_classes.items():
        dict_to_txt.write(str(k) + ','+ str(v) + '\n')
    dict_to_txt.close()

dataset_path = 'datasets/sportsMOT_volley_starter_pack/sportsMOT_volley_light_dataset'
txt_outpath = 'classificator.txt'

load_previous_markup = True
write_classes_to_txt = True
img_classes = markup(dataset_path, txt_outpath, load_previous_markup, write_classes_to_txt)