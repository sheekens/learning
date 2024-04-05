import os
import numpy as np
import cv2
from varname.helpers import debug
from operator import countOf
from dataloader.dataloader_sportsMOT import load_img_paths

keyboard_classes_mapping = {
    110: 'moving',
    109: 'static'
    }

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

def markup(dataset_path, txt_outpath):
    programm_finish = False
    img_paths = load_img_paths(dataset_path)
    if os.path.exists(txt_outpath):
        img_classes = load_classes_from_txt(txt_outpath)
        if not img_paths.keys() == img_classes.keys():
            for key in img_paths.keys():
                if not key in img_classes:
                    img_classes[key] = 'None'
            # print(img_classes)
    else:
        img_classes = dict.fromkeys(img_paths.keys(), None)
    
    for k in list(img_classes):
        if k not in img_paths.keys():
            del img_classes[k]

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
        cur_img = cv2.putText(
            cur_img,
            str('marked {} of {}'.format((len(img_classes) - countOf(img_classes.values(), 'None')), len(img_classes))),
            (10, 50),
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
        save_to_txt(img_classes, txt_outpath)
        if cur_img_id not in img_paths.keys():
            if cur_img_id > max(img_paths.keys()):
                cur_img_id = min(img_paths.keys())
            if cur_img_id < min(img_paths.keys()):
                cur_img_id = max(img_paths.keys())
    return img_classes

# TODO вынести в tools, добавив флаг на чтение из txt. переименовать точнее
def save_to_txt(img_classes, txt_outpath):
    dict_to_txt = open(txt_outpath, 'w')
    for k, v in img_classes.items():
        dict_to_txt.write(str(k) + ','+ str(v) + '\n')
    dict_to_txt.close()

dataset_path = 'testdata/sportsMOT_volley_starter_pack/sportsMOT_volley_light_dataset'
txt_outpath = 'tools/classificator.txt'

load_previous_markup = True
write_classes_to_txt = True
img_classes = markup(dataset_path, txt_outpath)

debug(img_classes)
# debug(img_paths)