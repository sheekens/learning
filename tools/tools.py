import os
import pathlib
import torch
from torch.functional import Tensor

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