import os
import cv2
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


def batch_from_path(image_path):
    """
    Input - путь до картинки
    Output - тензор с размером [1, 3, H, W]
    """
    input_image = cv2.imread(image_path)
    input_tensor = torch.from_numpy(input_image)
    input_tensor = input_tensor.permute(2, 0, 1) # CHW
    input_tensor = input_tensor.unsqueeze(0) # 1CHW
    input_tensor = input_tensor.to(torch.float32)
    input_tensor = input_tensor  / 255.0
    return input_tensor


def xywh2x1y1x2y2(xywh_bbox: tuple):
    return (
            xywh_bbox[0],
            xywh_bbox[1],
            xywh_bbox[0] + xywh_bbox[2],
            xywh_bbox[1] + xywh_bbox[3]
        )

    