import os
import cv2
from varname.helpers import debug
from player_tracking import load_gt, load_img_paths, xywh2x1y1x2y2

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

def crop_snippets(match_path:str , outdir: str, square: bool):
    outdir = os.path.abspath(outdir)
    match_name = os.path.basename(match_path)
    match_outdir = os.path.join(outdir, match_name)
    os.makedirs(match_outdir, exist_ok=True)
    img_paths = load_img_paths(match_path)
    gt_objects = load_gt(match_path)
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
        img_ext = os.path.splitext(img_path)[1]
        if square:
            img_ext = '_square' + img_ext
        img2draw = img.copy()
        for player_id, player_bbox in frame_objects.items():
            player_dirname = 'player_{}'.format(player_id)
            player_outdir = os.path.join(match_outdir, player_dirname)
            os.makedirs(player_outdir, exist_ok=True)
            cv_bbox = xywh2x1y1x2y2(player_bbox)
            cv_bbox_to_crop = cv_bbox
            if square:
                cv_bbox_to_crop = square_from_rectangle(player_bbox)
            snippet_name = '{}_{:02d}_{:06d}_x1_{}_y1_{}_x2_{}_y2_{}{}'.format(
                match_name,
                player_id,
                frame_number,
                cv_bbox[0],
                cv_bbox[1],
                cv_bbox[2],
                cv_bbox[3],
                img_ext
            )
            snippet_outpath = os.path.join(player_outdir, snippet_name) 
            player_snippet = img2draw[
                cv_bbox_to_crop[1]:cv_bbox_to_crop[3],
                cv_bbox_to_crop[0]:cv_bbox_to_crop[2],
                :
            ]
            debug(cv_bbox_to_crop)
            cv2.imshow('player_snippet_{}'.format(player_id), player_snippet)
            cv2.waitKey(20)
            cv2.destroyAllWindows()
            cv2.imwrite(snippet_outpath, player_snippet)
            print('snippet written to {}'.format(snippet_outpath))
if name == 'main' :
    outdir = 'output'
    match_path = r'C:\Users\user\Desktop\ml_course\projects\lesson1\datasets\sportsMOT_volley_starter_pack\sportsMOT_volley_light_dataset'
    crop_snippets(match_path, outdir, True)