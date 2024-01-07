import os
import torch
from torch import nn
from torch.functional import Tensor
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2
from varname.helpers import debug
from torch_convolution import manual_transform_to_tenzor
import random

class Simple2DConv(nn.Module):
    def __init__(
            self,
            ):
        super().__init__()
        # batch norm  ( statefull )
        self.bn1 = nn.BatchNorm2d(
                num_features=64
                )
        self.bn2 = nn.BatchNorm2d(
                num_features=128
                )
        # svertka (stetfull)
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

    def forward(self, x: Tensor): 
        x = self.conv1.forward(x)
        # debug(x.shape, prefix='conv1')
        x = self.bn1.forward(x)
        x = self.activation.forward(x)
        x = self.conv2.forward(x)
        # debug(x.shape, prefix='conv2')
        x = self.bn2.forward(x)
        x = self.activation.forward(x)
        return x

    

if __name__ == '__main__' :
    img_path = 'output\sportsMOT_volley_light_dataset\player_0\sportsMOT_volley_light_dataset_00_000001_x1_696_y1_442_x2_748_y2_565_square.jpg'

    img = cv2.imread(img_path)
    debug(type(img))
    img_tensor = torch.from_numpy(img)
    # #HWC to [CHW] default and to (0,1)
    # transform = torchvision.transforms.ToTensor()
    # img_torch_tensor = transform(img)

    # #HWC to [CHW] our and to (0,1)
    img_torch_tensor = manual_transform_to_tenzor(img_tensor)
    debug(img_tensor.shape, type(img_tensor))
    debug(img_torch_tensor.shape)

    stupid_conv_model = Simple2DConv()
    debug(img.shape)
    debug(img_torch_tensor.shape)
    out = stupid_conv_model.forward(img_torch_tensor)
    debug(out.shape)

    channels = (out.shape[1])
    debug(channels)
    resulting_shit = np.zeros((119,119,1))
    stack_counter = 0
    for chan in random.sample(range(1, channels), 16):
        convert = out[:,chan-1:chan,:,:]
    # 1 отрезать "служебный канал" в out
        convert = convert.squeeze(0)
	# 2 менять chw на hwc
        convert = convert.permute(1, 2, 0) # HWC
	# 3 переводить в numpy
	# 4 умножать на 255
        convert = (convert.detach().cpu().numpy()) * 255
	# 5 переводить в uint8
        convert = convert.astype(np.uint8)
        stack_counter += 1
        if stack_counter == 1:
            resulting_shit1 = convert
            continue
        if 1 < stack_counter <= 4:
             resulting_shit1 = np.hstack((convert, resulting_shit1))
        if stack_counter == 5:
            resulting_shit2 = convert
            continue
        if 5 < stack_counter <= 8:
             resulting_shit2 = np.hstack((convert, resulting_shit2))
        if stack_counter == 9:
            resulting_shit3 = convert
            continue
        if 9 < stack_counter <= 12:
             resulting_shit3 = np.hstack((convert, resulting_shit3))
        if stack_counter == 13:
            resulting_shit4 = convert
            continue
        if stack_counter > 13:
             resulting_shit4 = np.hstack((convert, resulting_shit4))
    resulting_shittest = np.vstack((resulting_shit1, resulting_shit2, resulting_shit3, resulting_shit4))
    cv2.imshow('aaa', resulting_shittest)
    cv2.waitKey(-1)