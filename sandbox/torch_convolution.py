import torch
from torch.functional import Tensor
import torchvision
import numpy as np
import cv2
from varname.helpers import debug
from tools.tools_img import img_transform_to_tenzor

### Svertochniy sloi
### Convolution layer


def softmax(tensor: Tensor):
    tensor_exp = np.exp(tensor)
    sum_exp = tensor_exp.sum()
    result = tensor_exp / sum_exp
    return result

if __name__ == '__main__' : 
    
    ### Functsiya activatsioi 
    ### Activation function
    input_tensor = torch.rand(
        size= (1,1, 3)
    ).uniform_(0,10)
    debug(input_tensor)


    # Normalization 
    batch_norm_layer = torch.nn.BatchNorm1d(num_features=1)
    norm_input_tensor = batch_norm_layer(input_tensor)
    # debug(norm_input_tensor)

    # Lineyniy viprimitel 
    activation_layer = torch.nn.Softmax(dim = 1)
    output_tensor = activation_layer(input_tensor)
    output_tensor_manual = softmax(input_tensor)
    print(output_tensor)
    debug(output_tensor_manual)
    

    img_path = 'output\sportsMOT_volley_light_dataset\player_0\sportsMOT_volley_light_dataset_00_000001_x1_696_y1_442_x2_748_y2_565_square.jpg'
    # img_path = './output/snippets/match_001_short/player_0/match_001_short_00_000021_x1_790_y1_443_x2_910_y2_563.jpg';
    img = cv2.imread(img_path)
    img_tensor = torch.from_numpy(img)
    transform = torchvision.transforms.ToTensor()
    img_torch_tensor = transform(img)

    
    conv = torch.nn.Conv2d(
        # 3 canal image
        in_channels=3,
        out_channels=16,
        kernel_size=(9,9),
        )
    params = conv.weight
    debug(params.shape)
    out = conv(img_torch_tensor)

    ## Same thing as just (conv)
    out = conv.forward(img_torch_tensor)


###  Conspect
# 1. Svertka 
# 2. Activatsiya 
# 3. Normalizatsiya
##No ne obezatelno v takom poryadke