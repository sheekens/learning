import os 
import numpy as np 
import cv2
from typing import List, Dict
from tools.tools_img import run_dataset

if __name__ == '__main__' :
    outdir = 'output/output_sportsMOT_volley_starter_pack'
    run_dataset('testdata/sportsMOT_volley_starter_pack/sportsMOT_volley_light_dataset', outdir)