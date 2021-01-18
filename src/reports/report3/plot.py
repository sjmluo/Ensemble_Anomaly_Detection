import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from src.CVHelper import *
from src.cvtest import *
from src.train import confusionmat

locs = ['test3/14','test3/15','test4/00','test4/01','test4/02','test4/03']

saves = []
print(os.getcwd())


for loc in locs:
    with open(f'src/reports/{loc}/save.pkl', 'rb') as f:
        saves.append(pickle.load(f).attr)

y_pred = {}
y_true = {}

for l in locs:
    y_pred[l] = []
    y_true[l] = []


for l, save in zip(locs, saves):
    for k in save['results']:
        if k.startswith('run'):
            y_pred[l].append(save['results'][k]['post']['y_pred'])
            y_true[l].append(save['results'][k]['post']['y_true'])
    y_pred[l] = np.concatenate(y_pred[l])
    y_true[l] = np.concatenate(y_true[l])


""" print(saves[0]['results'].keys())
print(saves[0]['results']['run1'].keys()) """

with open('src/reports/points.pkl', 'wb') as f:
    pickle.dump([y_pred, y_true], f)