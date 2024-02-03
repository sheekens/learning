# sheekens home py .\polar_dataloader.py --dataset_path D:\testing\learning\datasets\POLAR_dataset_100
# sheekens work py .\polar_dataloader.py --dataset_path C:\learning\learning\datasets\POLAR_dataset_100
# python current_files\polar_dataloader.py --dataset_path C:\cod\datasets\POLAR_dataset_100
import os
import torch
import cv2
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
import json
from pprint import pprint
from varname.helpers import debug
from cpor_snippets import square_from_rectangle
import pathlib

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument(
        "--is_debug", default=0, type=int, help="enable fast run on several samples"
    )
    return parser.parse_args()

# def square_snippet(img, bbox_x1y1x2y2, is_show=False):
#     x1,y1,w,h = bbox_x1y1x2y2_to_xywh(bbox_x1y1x2y2)
#     max_side = max(w,h)
#     snippet = np.zeros(shape=(max_side, max_side, 3),
#                        dtype=img.dtype)
#     snippet[:,:,1].fill(255)
#     snippet_x1y1x2y2 = square_from_rectangle((x1,y1,w,h))
#     snippet_from_img = img[
#         snippet_x1y1x2y2[1]:snippet_x1y1x2y2[3],
#         snippet_x1y1x2y2[0]:snippet_x1y1x2y2[2],
#         :
#     ]
#     snippet[
#         :snippet_from_img.shape[0],
#         :snippet_from_img.shape[1],
#         :
#     ] = snippet_from_img
#     if is_show:
#         cv2.imshow('snippet', snippet)
#         cv2.waitKey(-1)
#         cv2.destroyAllWindows()
#     return snippet

def square_snippet(img, bbox_x1y1x2y2, is_show=False):
    x1,y1,w,h = bbox_x1y1x2y2_to_xywh(bbox_x1y1x2y2)
    max_side = max(w,h)
    snippet = np.zeros(shape=(max_side, max_side, 3),
                       dtype=img.dtype)
    snippet[:,:,1].fill(255)
    snippet_x1y1x2y2 = square_from_rectangle((x1,y1,w,h))
    snippet_from_img = img[
        snippet_x1y1x2y2[1]:snippet_x1y1x2y2[3],
        snippet_x1y1x2y2[0]:snippet_x1y1x2y2[2],
        :
    ]
    # cv2.imshow('snippet', snippet_from_img)
    # cv2.waitKey(-1)
    # cv2.destroyAllWindows()
    # exit()
    # snippet[
    #     :snippet_from_img.shape[0],
    #     :snippet_from_img.shape[1],
    #     :
    # ] = snippet_from_img
    if w > h:
        top = (w-h)//2
        left = 0
        # debug(w,h,top,left)
    else:
        top = 0
        left = (h-w)//2
        # debug(w,h,top,left)
    bottom = top
    right = left
    # debug(bottom, right)
    borderType = cv2.BORDER_CONSTANT
    # value = [200,200,200]
    # snippet = img.copy()
    snippet = cv2.copyMakeBorder(snippet_from_img, top, bottom, left, right, borderType, value = [200,200,200])
    if is_show:
        debug(snippet.shape)
        cv2.imshow('snippet', snippet)
        cv2.waitKey(-1)
        cv2.destroyAllWindows()
    return snippet

def bbox_x1y1x2y2_to_xywh(bbox_x1y1x2y2):
    return [
        bbox_x1y1x2y2[0],
        bbox_x1y1x2y2[1],
        abs(bbox_x1y1x2y2[2]-bbox_x1y1x2y2[0]),
        abs(bbox_x1y1x2y2[1]-bbox_x1y1x2y2[3])
    ]

class PolarSnippets(Dataset):
    def __init__(self,dataset_path, square_img_size):
        self.dataset_path = dataset_path
        self.img_paths = []
        self.img_classes = []
        self.square_img_size = square_img_size
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if ".png" in file:
                    img_path = os.path.join(root, file)
                    img_class = os.path.basename(root)
                    self.img_paths.append(img_path)
                    self.img_classes.append(img_class)
        # debug(self.imgs_paths, self.img_classes)
        # debug(len(self.imgs_paths), len(self.img_classes))
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_class = self.img_classes[index]
        img = cv2.imread(img_path)
        square_img = cv2.resize(
            src=img,
            dsize=(self.square_img_size,self.square_img_size)
        )
        return square_img, img_class

class Polar_dataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.imgs_dir = os.path.join(self.dataset_path, "JPEGImages")
        self.annotations_dir = os.path.join(self.dataset_path, "Annotations")
        self.snippets_dir = os.path.join(self.dataset_path, 'snippets')
        self.imgs_paths = {}
        self.annotations = {}
        self.img_classes_dict = {}
        self.bndbox_dict = {}
        self.snippets_paths = {}
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if ".json" in file:
                    path_to_annotations = os.path.join(root, file)
                    with open(path_to_annotations, "r") as annotations_file:
                        annotations_dict = json.load(annotations_file)
                    img_name = annotations_dict["filename"]
                    path_to_img = os.path.join(self.imgs_dir, img_name)
                    if os.path.exists(path_to_img):
                        key2save = os.path.splitext(file)[0]
                        self.imgs_paths[key2save] = path_to_img
                        self.annotations[key2save] = annotations_dict
                    
                    for img_class in annotations_dict['persons'][0]['actions'].items():
                        if img_class[1] == 1:
                            self.img_classes_dict[key2save] = img_class[0]
                    raw_bbox = annotations_dict['persons'][0]['bndbox']
                    self.bndbox_dict[key2save] = [
                        raw_bbox['xmin'],
                        raw_bbox['ymin'],
                        raw_bbox['xmax'],
                        raw_bbox['ymax'],
                    ]
        if not os.path.exists(self.snippets_dir):
            os.makedirs(self.snippets_dir, exist_ok=True)
        self.crop_snippets()

    def visualize_bboxes(self):
        for img_name in self.annotations.keys():
            img_path = self.imgs_paths[img_name]
            img = cv2.imread(img_path)
            img2draw = img.copy()
            img2draw=cv2.putText(
                img2draw,
                str(self.img_classes_dict[img_name]),
                (10,25),
                1,
                1.7,
                (200,100,240),
                2
                )
            cv2.rectangle(
                    img2draw, 
                    #top left 
                    pt1=(self.bndbox_dict[img_name][0], self.bndbox_dict[img_name][1]),
                    #bottom rigth
                    pt2=(self.bndbox_dict[img_name][3], self.bndbox_dict[img_name][2]),
                    color=(200,100,240),
                    thickness=2
                    )
            cv2.imshow("img", img2draw)
            cv2.waitKey(-1)
    
    def crop_snippets(self):
        for img_name in self.annotations.keys():
            img_path = self.imgs_paths[img_name]
            img = cv2.imread(img_path)
            snippet = img.copy()
            # print(img_name)
            squared_snippet = square_snippet(snippet, self.bndbox_dict[img_name], is_show=False)
            snippet_class_path = os.path.join(self.snippets_dir, self.img_classes_dict[img_name])
            ensure_folder(snippet_class_path)
            snippet_path = os.path.join(snippet_class_path, f'{img_name}.png')
            cv2.imwrite(snippet_path, squared_snippet)
            print(snippet_path)

def ensure_folder(dir_fname):
    if not os.path.exists(dir_fname):
        try:
            pathlib.Path(dir_fname).mkdir(parents=True, exist_ok=True)
        except PermissionError:
            print('Unable to create {} directory. Permission denied'.format(dir_fname))
            return False
    return os.path.exists(dir_fname)

if __name__ == "__main__":
    arguments = parse_arguments()
    dataset_path = arguments.dataset_path
    is_debug = arguments.is_debug

    # polar_dataset = Polar_dataset(dataset_path)
    polar_snippets_dataset = PolarSnippets(dataset_path, 228)
    polar_snippets_dataloader = DataLoader(
        dataset=polar_snippets_dataset,
        batch_size=2
    )
    for img_batch, label_batch in polar_snippets_dataloader:
        debug(img_batch.size(), label_batch)
        exit()
    # polar_dataset.visualize_bboxes()
