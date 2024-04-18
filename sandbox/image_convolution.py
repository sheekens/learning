import os
import numpy as np
import cv2
from varname.helpers import debug
from tools.tools_img import load_img_paths

dataset_path = 'testdata/sportsMOT_volley_starter_pack/sportsMOT_volley_light_dataset'
dataset_images = load_img_paths(dataset_path)

for image_path in dataset_images.values():
    img = cv2.imread(image_path)
    img2draw = img.copy()
    img2draw = cv2.resize(img2draw, None, fx=0.5, fy=0.5)
    kernel = np.array([ #edge detection
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])
    # kernel = np.array([ #sharpen
    #     [0, -1, 0],
    #     [-1, 5, -1],
    #     [0, -1, 0]
    # ])
    filtered_image = cv2.filter2D(img2draw, -1, kernel)
    out_image = np.vstack([img2draw, filtered_image])
    cv2.imshow('test', out_image)
    cv2.waitKey(-1)