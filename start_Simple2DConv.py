# python start_Simple2DConv.py --image_path /home/alex/Downloads/istockphoto-119454172-1024x1024.jpg

import cv2
import torch 
import torchvision
import argparse
from varname.helpers import debug
from model.simple_convolution_model import Simple2DConv
from tools.tools_img import batch_from_path


def pars_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    # parser.add_argument("--is_debug", default=0, type=int, help="enable quick run")
    return parser.parse_args()


# Function to load the model
def load_model(model_path):
    model = Simple2DConv()
    model.load_state_dict(torch.load(model_path))
    model.eval()  
    return model
   
if __name__ == '__main__':
    # Load the model
    model_path = "/home/alex/repositories/learning/Simple2DConv.0004.pt"
    model = load_model(model_path)

    arguments = pars_arguments()
    image_path = arguments.image_path
    # is_debug = arguments.is_debug

    input_tensor = batch_from_path(image_path)
    
    # Run the model
    output = model(input_tensor)
    debug(output)
    print(output)
