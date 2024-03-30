import os 
import numpy as np 
import cv2
from typing import List, Dict
from varname.helpers import debug

# def img_outdir_save(outdir_path, file_path):
#     os.makedirs(outdir_path, exist_ok=True)
#     img_out_path = os.path.join(outdir, os.path.basename(file_path))
#     cv2.imwrite(img_out_path, file_path)
#     print('out written to', os.path.abspath(img_out_path))

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


def run_dataset(dataset_path:str , outdir: str): 
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
        
    print(gt_objects.keys())
    print(len(gt_objects.keys()))

        ########## output save
        # os.makedirs(outdir, exist_ok=True)
        # img_out_path = os.path.join(outdir, os.path.basename(img_path))
        # cv2.imwrite(img_out_path, img2draw)
        # print('out written to', os.path.abspath(img_out_path))


def frame_diff(dataset_path, diff_outdir):
    img_paths = load_img_paths(dataset_path)
    gt_objects = load_gt(dataset_path)
    # img_diff = cv2.imread(img_paths[1], cv2.IMREAD_GRAYSCALE)
    img_diff = 0
    for frame_number in gt_objects.keys():
        try: 
            img_path = img_paths[frame_number]
        except KeyError:
            # print('fffff')
            continue
        try: 
            prev_img_path = img_paths[frame_number-1]
        except KeyError:
            # print('aoaoao')
            # debug(frame_number)
            # img_diff = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            continue
        
        cur_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        prev_img = cv2.imread(prev_img_path, cv2.IMREAD_GRAYSCALE)


        # cur_img = cv2.GaussianBlur(cur_img,(5,5),0)
        # prev_img = cv2.GaussianBlur(cur_img,(5,5),0)
        # ret1,cur_img = cv2.threshold(cur_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # ret2,prev_img = cv2.threshold(prev_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                
        # img_temp = cv2.subtract(cur_img, prev_img)
        # img_temp = cv2.absdiff(cur_img, prev_img) #nicer, finds people in motion
        img_temp = cv2.bitwise_xor(cur_img, prev_img)

        # img_temp = cur_img - prev_img
        img_diff += img_temp
        # img_diff[img_diff>1] = 255

        # img_diff = img_diff - cur_img


        # img_diff = prev_img - cur_img #nice


        
        # img_diff = 1 / (img_diff)

        # img_diff = prev_img - cur_img #nice
        # img_diff = 1/((cur_img) / (0.5*prev_img))  #nicest

        # img_diff[img_diff==(0,0,0)] = (255,255,255)

        
        ########## output save
        os.makedirs(diff_outdir, exist_ok=True)
        diff_out_path = os.path.join(diff_outdir, os.path.basename(img_path))
        cv2.imwrite(diff_out_path, img_diff)
        print('diff out written to', os.path.abspath(diff_out_path))

        # cv2.imshow('jeezus', img_diff)
        # cv2.waitKey(-1)
    
    return img_diff

def load_diff_paths(dataset_path, diff_outdir):
    diff_paths = {}
    for root, dirs, files in os.walk(diff_outdir):
        for file_name in files:
            if '.jpg' in file_name and len(file_name) == 10: 
                diff_number = int(file_name[:-4])
                diff_paths[diff_number] = os.path.join(root, file_name)
    return diff_paths

def mix_diff_images(dataset_path, diff_outdir, outdir_path):
    diff_paths = load_diff_paths(dataset_path, diff_outdir)
    mix_diff = 0
    for diff_number in diff_paths.keys():
        try: 
            cur_diff_path = diff_paths[diff_number]
        except KeyError:
            continue
        # try: 
        #     prev_diff_path = diff_paths[diff_number-1]
        # except KeyError:
        #     continue
        cur_diff = cv2.imread(cur_diff_path)
        # prev_diff = cv2.imread(prev_diff_path)
        # mix_diff = cur_diff - prev_diff
        mix_diff = mix_diff + cur_diff
        # img_outdir_save(outdir_path=outdir_path, file_path=cur_diff_path)
       
        ########## output save
        os.makedirs(mix_diff_images_outdir_path, exist_ok=True)
        mix_diff_out_path = os.path.join(mix_diff_images_outdir_path, os.path.basename(cur_diff_path))
        cv2.imwrite(mix_diff_out_path, mix_diff)
        print('mix diff out written to', os.path.abspath(mix_diff_out_path))

        # cv2.imshow('jeezus', mix_diff)
        # cv2.waitKey(-1)
    return mix_diff

def xywh2x1y1x2y2(xywh_bbox: tuple):
    return (
            xywh_bbox[0],
            xywh_bbox[1],
            xywh_bbox[0] + xywh_bbox[2],
            xywh_bbox[1] + xywh_bbox[3]
        )

if __name__ == '__main__' :
    outdir = 'datasets/output_sportsMOT_volley_starter_pack'
    mix_diff_images_outdir_path = 'datasets/mix_diff_img'
    diff_outdir = 'datasets/diff_sportsMOT_volley_starter_pack'
    dataset_path = 'datasets/sportsMOT_volley_starter_pack/sportsMOT_volley_light_dataset' 
    # run_dataset(dataset_path, outdir)
    frame_diff(dataset_path, diff_outdir)
    # mix_diff_images(dataset_path, diff_outdir, mix_diff_images_outdir_path)