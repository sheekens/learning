import os
import torch
import torchvision
import numpy as np
from polar_dataloader import PolarSnippets, parse_arguments
from torch.utils.data import Dataset, DataLoader
from simple_convolution_model import Simple2DConv
from varname.helpers import debug
from sklearn.metrics import confusion_matrix, accuracy_score
import time
# from evidently.metric_preset import ClassificationPreset
# from evidently.report import Report


# dataset_path = 'D:/testing/learning/datasets/POLAR_dataset_100' ##sheekens home
# dataset_path = 'C:/testing/learning/datasets/POLAR_dataset_100' ##sheekens work
dataset_path = 'C:/testing/learning/datasets/POLAR_dataset_train_1000_val_200' ##sheekens work
train_polar_snippets_dataset = PolarSnippets((dataset_path+'/train'), square_img_size=24)
# train_polar_snippets_dataset = PolarSnippets((dataset_path+'/val'), square_img_size=24)
train_polar_snippets_dataloader = DataLoader(
    dataset=train_polar_snippets_dataset,
    batch_size=2
)
val_polar_snippets_dataset = PolarSnippets(dataset_path+'/val', square_img_size=24)
val_polar_snippets_dataloader = DataLoader(
    dataset=val_polar_snippets_dataset,
    batch_size=2
)
model = Simple2DConv()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001
)
epochs = 2
classes_indexes_array = np.arange(0,len(train_polar_snippets_dataset.classes_names),1)
t1 = time.time()
accuracy = 0
cur_accuracy = 0
precision = {}
recall = {}
tp = []
fp = []
fn = []
pred_sum = 0
for epoch in range(epochs):
    print(f'current epoch: [{epoch}]')
    total_conf_matrix = None
    
    ### train

    for img_batch, label_batch in train_polar_snippets_dataloader:
        out = model(img_batch)
        out_probabilities = model.softmax(out)
        loss = loss_fn(out, label_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(out, 1)
        conf_matrix = confusion_matrix(label_batch, preds, labels=classes_indexes_array)
        if total_conf_matrix is None:
            total_conf_matrix = conf_matrix
        else:
            total_conf_matrix += conf_matrix
    print('train')
    print(total_conf_matrix)
    debug(out_probabilities)
    t2 = time.time()

    ### validation

    total_conf_matrix = None
    for img_batch, label_batch in val_polar_snippets_dataloader:
        out = model(img_batch)
        out_probabilities = model.softmax(out)
        # loss = loss_fn(out, label_batch)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        _, preds = torch.max(out, 1)
        # debug(preds)
        conf_matrix = confusion_matrix(label_batch, preds, labels=classes_indexes_array)
        if total_conf_matrix is None:
            total_conf_matrix = conf_matrix
        else:
            total_conf_matrix += conf_matrix
    row_num = 0
    for true_class in total_conf_matrix: # type: ignore
        col_num = 0
        for predicted_class in true_class:
            if len(fp) < col_num+1:
                fp.append(0)
            if len(tp) < col_num+1:
                tp.append(0)
            if len(fn) < col_num+1:
                fn.append(0)
            pred_sum += predicted_class
            if col_num == row_num:
                tp[col_num] += predicted_class
            else:
                fp[col_num] += predicted_class
                fn[row_num] += predicted_class
            col_num += 1
        row_num += 1
    print('validation')
    print(total_conf_matrix)
    debug(out_probabilities)
    # debug(fp)
    # debug(tp)
    # debug(fn)
    cur_accuracy = sum(tp) / pred_sum
    if cur_accuracy > accuracy:
        accuracy = cur_accuracy
        print(f'new accuracy reached! it is [{accuracy*100}%] by now\nsaving model of epoch [{epoch:04d}]')
        torch.save(model.state_dict(), f'{dataset_path}/Simple2DConv.{epoch:04d}.pt')
    for true_class in range(len(total_conf_matrix)): # type: ignore
        precision[true_class] = (tp[true_class] / (tp[true_class] + fp[true_class]))*100
        recall[true_class] = (tp[true_class] / (tp[true_class] + fn[true_class]))*100
    print('epoching done by [{:.02f}s]\n'.format((t2 - t1)))
    # debug(precision, recall)