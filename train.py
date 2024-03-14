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


dataset_path = 'D:/testing/learning/datasets/POLAR_dataset_100' ##sheekens home
# dataset_path = 'C:/testing/learning/datasets/POLAR_dataset_100' ##sheekens work
polar_snippets_dataset = PolarSnippets(dataset_path, square_img_size=24)
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
epochs = 9001
classes_indexes_array = np.arange(0,len(polar_snippets_dataset.classes_names),1)
# debug(classes_indexes_array)
# debug(polar_snippets_dataset.classes_names())
t1 = time.time()
accuracy = 0
precision = {}
recall = {}
tp = []
fp = []
fn = []
pred_sum = 0
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
        # debug(loss)
        # debug(out, out_probabilities)
        # debug(out.size())
        # exit()
    if epoch == 2:
        t2 = time.time()
        print('epoching done by [{:.02f}s]\n'.format((t2 - t1)))
        row_num = 0
        for true_class in total_conf_matrix:
            # debug(row_num)
            col_num = 0
            for predicted_class in true_class:
                # debug(predicted_class)
                if len(fp) < col_num+1:
                    fp.append(0)
                if len(tp) < col_num+1:
                    tp.append(0)
                if len(fn) < col_num+1:
                    fn.append(0)
                # debug(predicted_class)
                # debug(col_num)
                pred_sum += predicted_class
                if col_num == row_num:
                    tp[col_num] += predicted_class
                ##TODO: make omerica great again or
                ##remake tp as fp by classes +
                ##add fn as fp by classes
                if col_num != row_num:
                    fp[col_num] += predicted_class
                    fn[row_num] += predicted_class
                col_num += 1
            row_num += 1
        print(total_conf_matrix)
        debug(fp)
        debug(tp)
        debug(fn)
        accuracy = sum(tp) / pred_sum
        for true_class in range(len(total_conf_matrix)):
            precision[true_class] = (tp[true_class] / (tp[true_class] + fp[true_class]))*100
            recall[true_class] = (tp[true_class] / (tp[true_class] + fn[true_class]))*100
        print('accuracy [{:.02f}%]\n'.format((accuracy*100)))
        debug(precision, recall)
        exit()