import pandas as pd
import numpy as np

from torchmetrics import ConfusionMatrix
import torchmetrics
import torch

import torchvision
import torchvision as transforms


import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC



#S1,S2,S3,1,2,3
#cutting off the letter S to create a list and to put in in a tensor after

df= pd.read_csv ('/mnt/Carla/Projects/Osama/Hyperparameter_tuning_results/LLovet_training/HOSHIDA_features-xiyue_bs=64_lr=0.001_nfolds=3_bagsize=512/Deployment/patient-preds.csv')

tru= df['HOSHIDA']
preds=df['pred']


def cut_s(tru, preds):
    to_return_tru = []
    to_return_preds = []
    for element in tru:
        x = (element.replace('S', ''))
        to_return_tru.append(int(x)-1)
    for element in preds:
        x = (element.replace('S', ""))
        to_return_preds.append(int(x)-1)
    return to_return_tru, to_return_preds

to_return_tru,to_return_preds = cut_s(tru,preds)
cut_s(tru, preds)

#print(to_return_tru)
#print(len(to_return_tru))

#print(to_return_preds)
#print(len(to_return_preds))
#already created the list beforehand, creating a tensor now 


to_return_preds= torch.tensor(to_return_preds)
to_return_tru= torch.tensor(to_return_tru)
#print the tensor
print(to_return_preds)
print(to_return_tru)

M=ConfusionMatrix(task="multiclass", num_classes=3)
print(M(to_return_preds, to_return_tru))


# creating a GUI









