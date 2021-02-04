import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from src.CVHelper import *
from src.cvtest import *
from src.train import confusionmat, bestalpha
import argparse
import pandas as pd

def load(locs, yinp = False):
    saves = []
    for loc in locs:
        with open(f'{prefix}{loc}/save.pkl', 'rb') as f:
            saves.append(pickle.load(f).attr)
    
    y_pred = {}
    y_true = {}
    y_inp = {}

    for l in locs:
        y_pred[l] = []
        y_true[l] = []
        y_inp[l] = []


    for l, save in zip(locs, saves):
        for k in save['results']:
            if k.startswith('run'):
                y_pred[l].append(save['results'][k]['post']['y_pred'])
                y_true[l].append(save['results'][k]['post']['y_true'])
                y_inp[l].append(save['results'][k]['post']['y_inp'])
        y_pred[l] = np.concatenate(y_pred[l])
        y_true[l] = np.concatenate(y_true[l])
        y_inp[l] = np.concatenate(y_inp[l])
    
    if yinp:
        return y_inp, y_pred, y_true
    return y_pred, y_true

def loadheatmap(locs):
    saves = []
    for loc in locs:
        with open(f'{prefix}{loc}/save.pkl', 'rb') as f:
            saves.append(pickle.load(f).attr)

    x = {}
    y = {}
    z = {}

    for l in locs:
        x[l] = []
        y[l] = []
        z[l] = []

    for l, save in zip(locs, saves):
        for k in save['results']:
            if k.startswith('run'):
                x[l].append(save['results'][k]['post']['contour_x'])
                y[l].append(save['results'][k]['post']['contour_y'])
                z[l].append(np.mean(save['results'][k]['post']['contour_z'], 0))
        x[l] = np.mean(x[l],0)
        y[l] = np.mean(y[l],0)
        #print(z[l])
        z[l] = np.mean(z[l],0)
    
    return x,y,z

def roc(locs, prefix = '', saveloc = './', showplot = True, bins = 1000):
    """
    ROC plot
    """
    
    y_pred, y_true = load(locs)

    for m in locs:
        y_p, y_t = y_pred[m], y_true[m]
        y_p, y_t = np.expand_dims(y_p,-1), np.expand_dims(y_t,-1)
        y_p = np.nan_to_num(y_p)
        y_p = (y_p - y_p.min())/(y_p.max() - y_p.min())
        tprs = []
        fprs = []
        for b in range(bins+1):
            cm = confusionmat(y_p, y_t, conf = b/bins)
            tpr = cm[1,1]/(cm[1,1] + cm[0,1])
            fpr = cm[1,0]/(cm[0,0] + cm[1,0])
            tprs.append(tpr)
            fprs.append(fpr)
        plt.plot(fprs, tprs, label = f'{locs[m]}')
    plt.plot([0,1],[0,1], 'r--')
    plt.legend()

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve')

    plt.savefig(f'{saveloc}/plot.png')
    print(f"Saved plot in {saveloc}/plot.png")

    if showplot:
        plt.show()

def perf(locs, prefix = '', saveloc = './', showplot = True, alpha = 0.5):
    """
    2D plot of model predictions vs true class
    """

    y_inp, y_pred, y_true = load(locs, True)

    for l in locs:
        plt.figure()
        inp, pred, true = y_inp[l], y_pred[l], y_true[l]
    
        df = pd.DataFrame([*[inp[:, col] for col in range(inp.shape[1])], pred, true]).T
        df = df.sort_values(by=[1,2])
        df = df.groupby([0,1]).mean()
        df = df.reset_index()

        inp = df.iloc[:,[0,1]].to_numpy()
        pred = df.iloc[:,2].to_numpy()
        true = df.iloc[:,3].to_numpy()

        if barrier == 'auto':
            alpha, _ = bestalpha(np.expand_dims(true,-1), np.expand_dims(pred,-1))
        
        print(alpha)

        pred = np.where(np.array(pred) < alpha, 0, 1)

        plt.plot(inp[np.logical_and(true==0, pred == 1),0], inp[np.logical_and(true==0, pred == 1),1], '.r', label = 'FP')
        plt.plot(inp[np.logical_and(true==1, pred == 0),0], inp[np.logical_and(true==1, pred == 0),1], 'xr', label = 'FN')
        plt.plot(inp[np.logical_and(true==0, pred == 0),0], inp[np.logical_and(true==0, pred == 0),1], '.k', label = 'TN')
        plt.plot(inp[np.logical_and(true==1, pred == 1),0], inp[np.logical_and(true==1, pred == 1),1], 'xk', label = 'TP')
        plt.legend()
        plt.title(f"Plot of {locs[l]} with alpha {alpha}")

        plt.savefig(f'{saveloc}/{l}plot.png')
        print(f"Saved plot in {saveloc}/{l}plot.png")

        if showplot:
            plt.show()

def heatmap(locs, prefix = '', saveloc = './', showplot = True):
    x,y,z = loadheatmap(locs)
    for l in locs:
        plt.figure()
        #plt.contour([inp[:,0]], [inp[:,1]], np.expand_dims(pred, axis = -1), colors='black')
        plt.contourf(x[l], y[l], z[l], cmap='RdGy')
        plt.colorbar()

        plt.savefig(f'{saveloc}/{l}plot.png')
        print(f"Saved plot in {saveloc}/{l}plot.png")
        
        if showplot:
            plt.show()

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('plot_type', help = "Type of plot, ROC or performance(perf)", choices = ['roc', 'perf','heat'])
    parser.add_argument('--locs', action = 'store', nargs = '+', help = 'Directories containing save.pkl')
    parser.add_argument('--names', action = 'store', nargs = '*', help = 'Names for corresponding save')
    parser.add_argument('--prefix', help = 'prefix for directory for all saves', default = '')
    parser.add_argument('--saveloc', help = 'Location to save plot', default = './')
    parser.add_argument('-q', '--quiet', help = 'Disable show plot', action = 'store_true', default = False)
    parser.add_argument('-s', '--steps', help = 'Number of steps or bins ROC should take', default = 100, type = int)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-f', type = float, help = 'Value where predictions are lower than are categorised as class 0', default = 0.5)
    group.add_argument('-a', '--auto', action='store_true', help = 'Automatic value for best F1', default = False)

    args = parser.parse_args()

    locs = args.locs
    names = args.names
    prefix = args.prefix

    if names == None:
        names = locs

    if len(locs) != len(names):
        raise ValueError(f"Length of locs {len(locs)}, and length of names {len(names)} are not equal.")

    if '.' in args.saveloc.split('/')[-1]:
        raise ValueError("Bad directory")

    plotargs = {}

    barrier = 'auto' if args.auto else args.f

    for l, n in zip(locs, names):
        plotargs[l] = n

    if args.plot_type == 'roc':
        roc(plotargs, prefix, args.saveloc, not args.quiet, args.steps)
    elif args.plot_type == 'perf':
        perf(plotargs, prefix, args.saveloc, not args.quiet, barrier)
    elif args.plot_type == 'heat':
        heatmap(plotargs, prefix, args.saveloc, not args.quiet)
    else:
        raise ValueError('Bad arguement')
