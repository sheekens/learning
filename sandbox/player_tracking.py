# TODO вынести общие функции в tools
# TODO решить проблему с testdata\sportsMOT_volley_starter_pack\sportsMOT_volley_light_dataset\img1 (deleted by us)

import os 
import numpy as np 
import cv2
from typing import List, Dict
from varname.helpers import debug
from dataloader.dataloader_sportsMOT import load_gt, load_img_paths
from tools.tools import xywh2x1y1x2y2

def run_dataset(dataset_path:str , outdir: str): 
    img_paths = load_img_paths(dataset_path)
    gt_objects = load_gt(dataset_path)
    max_player_id = 0
    for players in gt_objects.values():
        if max(players.keys()) > max_player_id:
            max_player_id = max(players.keys())
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
                    color=[player_id],
                    thickness=2)

        os.makedirs(outdir, exist_ok=True)
        img_out_path = os.path.join(outdir, os.path.basename(img_path))
        cv2.imwrite(img_out_path, img2draw)
        print('out written to', os.path.abspath(img_out_path))

if __name__ == '__main__' :
    outdir = 'output'
    run_dataset('c:\cod\datasets', outdir)