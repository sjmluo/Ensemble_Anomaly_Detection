import pickle
import os
import time
import numpy as np
import random
from VAE import VAE
from train import confusionmat, kfoldstratify

class Save:
    def __init__(self, wdir, attr = None):
        self.wdir = wdir
        if attr is not None:
            self.saveargs(attr)

    def saveargs(self, attr):
        self.attr = attr.copy()

        todrop = ['inp', 'labels', 'testdata', 'save', 'model']
        
        for name in todrop:
            if name in self.attr:
                self.attr.pop(name, None)
        
    def loadargs(self):
        return self.attr
    
    def save(self):
        with open(f'{self.wdir}/save.pkl', 'rb') as file: 
            pickle.dump(self, file)
    
    def load(self):
        with open(f'{self.wdir}/save.pkl', 'rb') as file: 
            self = pickle.load(file)
    
    def commit(self, attr):
        self.saveargs(attr)
        self.save()
    
    def rollback(self):
        self.load()
        return self.loadargs()

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class CVHelper:
    def __init__(self, vaeArgs, data, compile_func, name, cvRuns = 3,
                description = None, 
                epochs=500, 
                k = 10, 
                seed = 0, 
                verbose = 0, 
                testsplit = (1, 20), 
                wdir = 'src/reports/test1',
                **kwargs):
        wdir = f'{wdir}/{name}'

        self.save = Save(wdir)
        self.data = data
        self.compile_func = compile_func
        self.description = description
        self.name = name
        self.progress = [0,0] #CV, k

        try:
            os.mkdir(wdir)
        except FileExistsError:
            self.loadArgs()
            return

        self.vaeArgs = vaeArgs

        if not isinstance(data, callable):
            _data = data
        else:
            _data = data()
        
        (inp, labels), testdata = _data
        
        self.inp = inp
        self.labels = labels
        self.testdata = testdata
        self.epochs = epochs
        self.k = k
        self.cvRuns = cvRuns
        self.seed = seed
        self.verbose = verbose
        self.testsplit = testsplit
        self.wdir = wdir
        self.kwargs = kwargs

    def saveArgs(self):
        self.save.commit(self.__dict__)

    def saveCheck(self, incattr):
        if incattr['description'] != self.description:
            return False
        if incattr['data'] != self.data:
            return False
        if incattr['compile_func'] != self.compile_func:
            return False
        return True

    def loadArgs(self):
        self.save.load()
        incattr = self.save.rollback()

        if not self.saveCheck(incattr):
            raise ValueError(f'Name {self.name} is invalid, save already exists')

        if not isinstance(self.data, callable):
            _data = self.data
        else:
            _data = self.data()

        (inp, labels), testdata = _data
        
        self.inp = inp
        self.labels = labels
        self.testdata = testdata
    
    def createModel(self):
        self.model = VAE(**self.vaeArgs)
        self.model.addcompile(self.compile_func)
        self.model._compile()
    
    def crossvalidation(self):
        results = {'acc':[], 'loss':[], 'cm':[], 'spec':[]}
        if k == 0: return results

        for i in range(self.progress[0], self.cvRuns):
            set_seed(seed)
            ind = kfoldstratify(labels.iloc[:,-1], self.k)
            for j in range(self.progress[1], self.k):
                


        return results

    def fold(self, trainin, trainout, testin, testout, results):
        default_kwargs = {'x':trainin, 'y':trainout, 'epochs': self.epochs, 'verbose': self.verbose, 'validation_data':(testin, testout)}
        default_kwargs.update(kwargs)
        default_kwargs['callbacks'] = kwargs['callbacks'](f'{wdir}/logs/cv{i}')

        hist = model.fit(**default_kwargs).history

        pred = model.predict(ktestin)
        cm = confusionmat(pred[-1], ktestout[-1])
        results['loss'].append([hist[l][-1] for l in loss_record])
        results['acc'].append((cm[0][0] + cm[1][1])/cm.sum())
        results['spec'].append(cm[1][1]/(cm[:,1].sum()))
        results['cm'].append(cm)
