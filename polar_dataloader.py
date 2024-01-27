# py .\polar_dataloader.py --dataset_path D:\testing\learning\datasets\POLAR_dataset_100
# python current_files\polar_dataloader.py --dataset_path C:\cod\datasets\POLAR_dataset_100
import os
import torch
import cv2
import numpy as np
import argparse
from torch.utils.data import Dataset
import json
from pprint import pprint
from varname.helpers import debug

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument(
        "--is_debug", default=0, type=int, help="enable fast run on several samples"
    )
    return parser.parse_args()

class Polar_dataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.imgs_dir = os.path.join(self.dataset_path, "JPEGImages")
        self.annotations_dir = os.path.join(self.dataset_path, "Annotations")
        self.imgs_paths = {}
        self.annotations = {}
        self.img_classes_dict = {}
        self.bndbox_dict = {}
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

                    for bndbox in annotations_dict['persons'][0]['bndbox'].items():
                        try:
                            self.bndbox_dict[key2save].append(bndbox[1])
                        except KeyError:
                            self.bndbox_dict[key2save]=[]
                            self.bndbox_dict[key2save].append(bndbox[1])


if __name__ == "__main__":
    arguments = parse_arguments()
    dataset_path = arguments.dataset_path
    is_debug = arguments.is_debug

    polar_dataset = Polar_dataset(dataset_path)

    for img_name in polar_dataset.annotations.keys():
        img_path = polar_dataset.imgs_paths[img_name]
        img = cv2.imread(img_path)
        img2draw = img.copy()
        img2draw=cv2.putText(
            img2draw,
            str(polar_dataset.img_classes_dict[img_name]),
            (10,25),
            1,
            1.7,
            (200,100,240),
            2
            )
        cv2.rectangle(
                img2draw, 
                #top left 
                pt1=(polar_dataset.bndbox_dict[img_name][0], polar_dataset.bndbox_dict[img_name][1]),
                #bottom rigth
                pt2=(polar_dataset.bndbox_dict[img_name][3], polar_dataset.bndbox_dict[img_name][2]),
                color=(200,100,240),
                thickness=2
                )
        cv2.imshow("img", img2draw)
        cv2.waitKey(-1)