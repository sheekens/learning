import os
import torch
from torch import nn
from torch.functional import Tensor
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2
from varname.helpers import debug
from tools.tools_img import batch_from_path
import random

class Simple2DConv(nn.Module):
    def __init__(
            self,
            ):
        super().__init__()
        # batch norm (statefull)
        self.bn1 = nn.BatchNorm2d(
                num_features=64
                )
        self.bn2 = nn.BatchNorm2d(
                num_features=128
                )
        # svertka (statefull)
        self.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=(5, 5),
                )
        self.conv2 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(1, 1),
                )
        # aktivatsiya (stateless)
        self.activation = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.classes_number = 9
        self.classification_head = nn.Linear(128, self.classes_number)
        self.softmax = nn.Softmax(1)

    def forward(self, x: Tensor): 
        x = self.conv1(x)
        x = self.bn1.forward(x)
        x = self.activation(x)
        x = self.conv2.forward(x)
        x = self.bn2.forward(x)
        x = self.activation.forward(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classification_head(x)
        return x


if __name__ == '__main__' :
    img_path = 'output\sportsMOT_volley_light_dataset\player_0\sportsMOT_volley_light_dataset_00_000001_x1_696_y1_442_x2_748_y2_565_square.jpg'
    img = cv2.imread(img_path)
    debug(img.shape)
    exit()
    img_tensor = torch.from_numpy(img)
    # #HWC to [CHW] our and to (0,1)
    img_torch_tensor = batch_from_path(img_tensor)
    stupid_conv_model = Simple2DConv()
    out = stupid_conv_model.forward(img_torch_tensor)
    channels = (out.shape[1])
    converted_imgs_list = []
    for chan in random.sample(range(1, channels), 16):
        converted_img = out[:,chan-1:chan,:,:]
    # 1 отрезать "служебный канал" в out
        converted_img = converted_img.squeeze(0)
	# 2 менять chw на hwc
        converted_img = converted_img.permute(1, 2, 0) # HWC
	# 3 переводить в numpy
	# 4 умножать на 255
        converted_img = (converted_img.detach().cpu().numpy()) * 255
	# 5 переводить в uint8
        converted_img = converted_img.astype(np.uint8)
        converted_imgs_list.append(converted_img)
    
    ## вынести в аргумент функции!!!!
    row_len = 4

    stack_amount = int(len(converted_imgs_list) / row_len)
    rows_list = []
    imgs_nums_list = []
    vstack_rows = ()
    previous = 0
    imgs_nums_list_counter = 0
    current = row_len
    converted_imgs_list_hstack = np.hstack((converted_imgs_list))

    for j in range(stack_amount):
        for i in range(row_len):
            rowed_imgs = np.hstack((converted_imgs_list[previous:current]))
            rows_list.append(rowed_imgs)
        cv2.imshow('aaa', rowed_imgs)
        cv2.waitKey(-1)
        imgs_nums_list.append(previous)
        vstack_rows = vstack_rows + (rows_list[imgs_nums_list[j]],)
        imgs_nums_list_counter += 1
        previous += row_len
        current += row_len
        
    stacked_rows = np.vstack(vstack_rows)
        
    cv2.imshow('aaa', stacked_rows)
    cv2.waitKey(-1)