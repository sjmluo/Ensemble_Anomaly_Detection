import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from src.CVHelper import *
from src.cvtest import *

locs = ['06','07','08']

saves = []
print(os.getcwd())




for loc in locs:
    with open(f'src/reports/test2/{loc}/save.pkl', 'rb') as f:
        saves.append(pickle.load(f).attr)

y_pred = {'06':[],'07':[],'08':[]}
y_true = {'06':[],'07':[],'08':[]}


for l, save in zip(locs, saves):
    for k in save['results']:
        if k.startswith('run'):
            y_pred[l].append(save['results'][k]['y_pred'])
            y_true[l].append(save['results'][k]['y_true'])
    y_pred[l] = np.concatenate(y_pred[l])
    y_true[l] = np.concatenate(y_true[l])


""" print(saves[0]['results'].keys())
print(saves[0]['results']['run1'].keys()) """



with open('src/reports/test2/report/points.pkl', 'wb') as f:
    pickle.dump([y_pred, y_true], f)