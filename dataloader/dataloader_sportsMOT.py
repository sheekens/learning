import os 
import numpy as np 
import cv2
from typing import List, Dict
from tools.tools_img import xywh2x1y1x2y2

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

def load_img_paths(dataset_path: str) -> Dict[int, str]: 
    img_paths = {}
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if '.jpg' in file_name and len(file_name) == 10: 
                frame_number = int(file_name[:-4])
                img_paths[frame_number] = os.path.join(root, file_name)
    return img_paths


def run_dataset(dataset_path:str, outdir:str): 
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

        os.makedirs(outdir, exist_ok=True)
        img_out_path = os.path.join(outdir, os.path.basename(img_path))
        cv2.imwrite(img_out_path, img2draw)
        print('out written to', os.path.abspath(img_out_path))
        # cv2.imshow('fff', img2draw)
        # cv2.waitKey(-1)
        # cv2.destroyAllWindows()

if __name__ == '__main__' :
    outdir = 'output/output_sportsMOT_volley_starter_pack'
    run_dataset('/home/alex/Telegram Desktop/sportsMOT_volley_starter_pack.002/sportsMOT_volley_starter_pack.002/sportsMOT_volley_light_dataset', outdir)