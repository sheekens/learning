import torch
from torch.functional import Tensor
import torchvision
import numpy as np
import cv2
from varname.helpers import debug

### Svertochniy sloi
### Convolution layer
### 



def softmax(tensor: Tensor):
    tensor_exp = np.exp(tensor)
    sum_exp = tensor_exp.sum()
    result = tensor_exp / sum_exp
    return result




def manual_transform_to_tenzor (input_tenzor) -> Tensor:

    ### 
    # Imread reads  images as HWC 
    # Torch reads images as CHW
    ###
    input_tenzor = input_tenzor.permute(2, 0, 1) # CHW
    input_tenzor = input_tenzor.unsqueeze(0) # 1CHW
    input_tenzor = input_tenzor.to(torch.float32)
    img_min = input_tenzor.min()
    img_max = input_tenzor.max()
    # img_tensor = (img_tensor - img_min) / torch.abs(img_max - img_min)
    input_tenzor = input_tenzor  / 255.0
    return input_tenzor



if __name__ == '__main__' : 
    
    ### Functsiya activatsioi 
    ### Activation function
    input_tensor = torch.rand(
        size= (1,1, 3)
    ).uniform_(0,10)
    debug(input_tensor)

    # debug(input_tensor.max())
    # debug(input_tensor.min())
    # debug(input_tensor.sum())

    # Normalization 
    batch_norm_layer = torch.nn.BatchNorm1d(num_features=1)
    norm_input_tensor = batch_norm_layer(input_tensor)
    debug(norm_input_tensor)
    # Lineyniy viprimitel 
    activation_layer = torch.nn.Softmax(dim = 1)
    output_tensor = activation_layer(input_tensor)
    output_tensor_manual = softmax(input_tensor)
    debug(output_tensor)
    debug(type(output_tensor[0][0]))
    debug(output_tensor_manual)
    debug(type(output_tensor_manual[0][0]))
    debug(output_tensor_manual.to(torch.float16) == output_tensor.to(torch.float16))
    # debug(output_tensor.max())
    # debug(output_tensor.min())
    # debug(output_tensor.sum())

    # To (0, 255) (optional)
    # output_tensor *= 255 
    # output_tensor = output_tensor.to(torch.uint8)
    # debug(output_tensor)
    # torch.manual_seed(420)
    # np.random.seed(420)
<<<<<<< HEAD
    img_path = 'output\sportsMOT_volley_light_dataset\player_0\sportsMOT_volley_light_dataset_00_000001_x1_696_y1_442_x2_748_y2_565_square.jpg';
=======
    img_path = './output/snippets/match_001_short/player_0/match_001_short_00_000021_x1_790_y1_443_x2_910_y2_563.jpg';
>>>>>>> fd4d273d23ee5bb5f12e81c2292b82a33d154f69
    img = cv2.imread(img_path)
    h = img.shape[0]
    w = img.shape[1]
    c = img.shape[2]
    img_tensor = torch.from_numpy(img)
    transform = torchvision.transforms.ToTensor()
    img_torch_tensor = transform(img)

    # debug(img_tensor == img_torch_tensor)
    # img_tensor = torch.rand(
    #         size = (1, 3, h, w)
    #         ).uniform_(-1,1)
    # debug(img_tensor)
    # debug(img_tensor.shape)
    # print(type(img_tensor))
    # debug(img_tensor.max())
    # debug(img_tensor.min())
    # debug(torch.__version__)
    conv = torch.nn.Conv2d(
        # 3 canal image
        in_channels=3,
        out_channels=16,
        kernel_size=(9,9),
        )
    params = conv.weight
    debug(params.shape)
    # debug(params)
    out = conv(img_torch_tensor)
    debug(out.shape)
    ## Same thing as just (conv)
    out = conv.forward(img_torch_tensor)
    debug(out.shape)


###  Conspect
# 1. Svertka 
# 2. Activatsiya 
# 3. Normalizatsiya
##No ne obezatelno v takom poryadke