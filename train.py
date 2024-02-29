import os
import torch
import torchvision
import numpy as np
from polar_dataloader import PolarSnippets, parse_arguments
from torch.utils.data import Dataset, DataLoader
from simple_convolution_model import Simple2DConv
from varname.helpers import debug


dataset_path = 'D:/testing/learning/datasets/POLAR_dataset_100'
polar_snippets_dataset = PolarSnippets(dataset_path, 24)
polar_snippets_dataloader = DataLoader(
    dataset=polar_snippets_dataset,
    batch_size=2
)
model = Simple2DConv()
epochs = 9000
for epoch in range(epochs):
    for img_batch, label_batch in polar_snippets_dataloader:
        debug(img_batch.size())
        debug(label_batch)
        out = model(img_batch)
        out_probabilities = model.softmax(out)
        debug(out, out_probabilities)
        debug(out.size())
        exit()