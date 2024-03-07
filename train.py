import os
import torch
import torchvision
import numpy as np
from polar_dataloader import PolarSnippets, parse_arguments
from torch.utils.data import Dataset, DataLoader
from simple_convolution_model import Simple2DConv
from varname.helpers import debug
from sklearn.metrics import confusion_matrix


dataset_path = 'D:/testing/learning/datasets/POLAR_dataset_100'
polar_snippets_dataset = PolarSnippets(dataset_path, 24)
polar_snippets_dataloader = DataLoader(
    dataset=polar_snippets_dataset,
    batch_size=2
)
model = Simple2DConv()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001
)
epochs = 9000
classes_indexes_array = np.arange(0,len(polar_snippets_dataset.classes_names),1)
# debug(classes_indexes_array)
# debug(polar_snippets_dataset.classes_names())
for epoch in range(epochs):
    total_conf_matrix = None
    for img_batch, label_batch in polar_snippets_dataloader:
        # debug(img_batch.size())
        # debug(label_batch)
        out = model(img_batch)
        # debug(out)
        out_probabilities = model.softmax(out)
        loss = loss_fn(out, label_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(out, 1)
        # debug(preds)
        conf_matrix = confusion_matrix(label_batch, preds, labels=classes_indexes_array)
        if total_conf_matrix is None:
            total_conf_matrix = conf_matrix
        else:
            total_conf_matrix += conf_matrix
        # debug(conf_matrix)
        # debug(loss)
        # debug(out, out_probabilities)
        # debug(out.size())
        # exit()
    debug(total_conf_matrix)
    exit()