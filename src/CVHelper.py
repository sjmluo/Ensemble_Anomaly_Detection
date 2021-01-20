import pickle
import os
import time
import numpy as np
import random
import tensorflow as tf
from src.VAE import VAE
from src.train import confusionmat, kfoldstratify


class Save:
    def __init__(self, wdir, attr = None):
        self.wdir = wdir
        if attr is not None:
            self.saveargs(attr)

    def save(self):
        with open(f'{self.wdir}/save.pkl', 'wb') as file: 
            pickle.dump(self, file)
    
    def saveargs(self, attr):
        self.attr = attr.copy()

        todrop = ['inp', 'labels', 'save', 'model']
        
        for name in todrop:
            if name in self.attr:
                self.attr.pop(name, None)
      
    def load(self):
        with open(f'{self.wdir}/save.pkl', 'rb') as file:
            pkl = pickle.load(file)
            self.__dict__.update(pkl.__dict__)
    
    def loadargs(self):
        return self.attr

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
    def __init__(self, modelArgs, data, compile_func, name, cvRuns = 3,
                description = None, 
                epochs=500, 
                k = 10, 
                seed = 0, 
                verbose = 0, 
                wdir = 'src/reports/test1',
                train_preprocessing = None,
                test_preprocessing = None,
                **kwargs):
        """
        kwargs:
        model: Class of model to be created
        postfold: Function with signature postfold(model, result), called after fitting on a fold
        writeResults: Function with signature writeResults(file, result), logs calculated metrics into file
        postresult: Function with signature postresult(file, result), allows for writing results to file.
         Similar to write results but called on results form postfold
        callbacks: function that returns callbacks
        """

        valid_kwargs = {'model', 'postfold', 'postresult','callbacks', 'writeresults'}
        
        for kw in kwargs:
            if kw not in valid_kwargs:
                raise ValueError(f'{kw} is not a valid kwarg')

        wdir = f'{wdir}/{name}'

        self.save = Save(wdir)
        self.data = data
        self.compile_func = compile_func
        self.description = description
        self.name = name
        self.progress = [0,0] #CV, k
        self.results = {}
        self.model = None

        try:
            os.mkdir(wdir)
        except FileExistsError:
            self.loadArgs()
            return

        self.model_class = VAE if 'model' not in kwargs else kwargs['model']
        self.modelArgs = modelArgs

        self.resetData()

        self.postfold = None if 'postfold' not in kwargs else kwargs['postfold']
        self.postresult = None if 'postresult' not in kwargs else kwargs['postresult']
        self._writeResults = writeResults if 'writeResults' not in kwargs else kwargs['writeResults']
        self.epochs = epochs
        self.k = k
        self.cvRuns = cvRuns
        self.seed = seed
        self.verbose = verbose
        self.wdir = wdir
        self.train_preprocessing = train_preprocessing
        self.test_preprocessing = test_preprocessing
        self.loss_record = ['loss']
        self.kwargs = kwargs
        self.trainingTime = 0

        self.saveArgs()

    def saveArgs(self):
        self.save.commit(self.__dict__)

    def saveCheck(self, incattr):
        if incattr['description'] != self.description:
            print(f'Description is different {incattr["description"]}, {self.description}')
            return False
        if incattr['data'] != self.data:
            print('Data is different')
            return False
        if self.compile_func != incattr['compile_func']:
            print(f'Compile function is different {incattr["compile_func"]}, {self.compile_func}')
            return False
        return True

    def resetData(self):
        print('called reset data')
        if not callable(self.data):
            _data = self.data
        else:
            _data = self.data()

        (inp, labels) = _data
        self.inp = inp
        self.labels = labels
        self.createModel()

    def loadArgs(self):
        incattr = self.save.rollback()

        if not self.saveCheck(incattr):
            raise ValueError(f'Name {self.name} is invalid, save already exists')
        
        self.__dict__.update(incattr)

        self.resetData()
    
    def createModel(self):
        if self.model == None:
            self.model = self.model_class(**self.modelArgs)
            self.model.addcompile(self.compile_func)
            self.model._compile()
        else:
            self.model.reset_model()
    
    def crossvalidation(self):
        for i in range(self.progress[0], self.cvRuns): #Every i indicates a new run
            if self.progress[1] == 0:
                if os.path.exists(f'{self.wdir}/run{self.progress[0]}'):
                    try:
                        os.rmdir(f'{self.wdir}/run{self.progress[0]}')
                    except OSError:
                        raise OSError(f'Folder {self.wdir}/run{self.progress[0]} already exists, please check and delete the folder')
                os.mkdir(f'{self.wdir}/run{self.progress[0]}')
            start = time.perf_counter()            
            set_seed(i)
            fold_ind = kfoldstratify(self.labels.iloc[:,-1], self.k)

            if self.progress[1] == 0:
                self.results[f'run{i}'] = {'acc':[], 'loss':[], 'cm':[], 'spec':[], 'y_pred': [], 'y_true': [], 'loglikelihood': [],
                    'totalTime':0, 'trainingTime':0}
            results = self.results[f'run{i}']

            if self.postfold is not None:
                results['post'] = {}

            results['totalTime'] += time.perf_counter() - start

            for j in range(self.progress[1], self.k):
                print(self.progress)
                start = time.perf_counter()
                self.model.reset_model()
                self.model.reset_states()
                arr = np.setdiff1d(list(range(self.k)), [i]).astype(int)
                trainind, testind = np.concatenate(fold_ind[arr]), fold_ind[i]
                trainin, trainout = self.inp.iloc[trainind, :].copy(), self.labels.iloc[trainind, :].copy()
                testin, testout = self.inp.iloc[testind,:].copy(), self.labels.iloc[testind,:].copy()
                
                if self.train_preprocessing is not None:
                    trainin, trainout = self.train_preprocessing(trainin, trainout)
                if self.test_preprocessing is not None:
                    testin, testout = self.test_preprocessing(testin, testout)

                self.fold(trainin, trainout, testin, testout, results)

                results['totalTime'] += time.perf_counter() - start
                self.progress[1] += 1
                results['trainingTime'] = self.trainingTime
                
                self.save.commit(self.__dict__)

            self.resetData()

            with open(f'{self.wdir}/run{self.progress[0]}/results', 'w') as file:
                self.writeResults(file, results)

            self.progress[0] += 1
            self.progress[1] = 0

            self.trainingTime = 0
        
        with open(f'{self.wdir}/overall', 'w') as file:
            inf = self.model.info()
            for a in inf:
                file.write(f'{a}: {inf[a]}\n')
            results = {'acc':[], 'loss':[], 'cm':[], 'spec':[], 'loglikelihood': [],
                    'totalTime':0, 'trainingTime':0}
            for key in results:
                lst = [self.results[run][key] for run in self.results]
                results[key] = np.mean(lst, 0)
            
            if self.postfold is not None:
                _results = {}
                for key in self.results['run0']['post']:
                    lst = [self.results[run]['post'][key] for run in self.results]
                    _results[key] = np.mean(lst, 0)
                results['post'] = _results
                print(_results)
                
            self.results['overall'] = results
            self.writeResults(file, results)
        self.save.commit(self.__dict__)
        return results

    def writeResults(self, file, results):
        if self.description is not None:
            file.write(f'{self.description}\n')
        self._writeResults(file, results)
        file.write('\n'.join([f"CV took {results['totalTime']} seconds",
            f"Pure training took {results['trainingTime']}\n"]))
        if self.postresult is not None:
            self.postresult(file, results['post'])

    def fold(self, trainin, trainout, testin, testout, results):
        default_kwargs = {'x':trainin, 'y':trainout, 'epochs': self.epochs, 'verbose': self.verbose, 'validation_data':(testin, testout)}
        default_kwargs['callbacks'] = self.kwargs['callbacks'](f'{self.wdir}/run{self.progress[0]}/logs/cv{self.progress[1]}')
        start = time.perf_counter()
        hist = self.model.fit(**default_kwargs).history
        self.trainingTime += time.perf_counter() - start

        pred = self.model.predict(testin)
        
        crunch_predictions(pred[-1], testout[-1], results)

        results['loss'].append([hist[l][-1] for l in self.loss_record])

        if self.postfold is not None: self.postfold(self.model, results['post'])


def writeResults(file, results):
    a = np.mean(results['cm'],axis = 0)
    ave = (a[0][0] + a[1][1])/a.sum()
    spec = a[0][0]/(a[:,0].sum())
    loss = np.nanmean(results['loss'],0)
    sensi = a[1][1]/a[:,1].sum()
    file.writelines('\n'.join([
        f"Average accuracy: {ave}", 
        f"Balanced acc: {(sensi+spec)/2}", 
        f"Average specificity: {spec}", 
        f"Average sensitivity (Detection rate): {sensi}",
        f"Average loss: {loss}", 
        f"Average False Alarm: {a[0,1]/a[:,1].sum()}", 
        f"Average F1: {2*a[1,1]/(2*a[1,1]+a[0,1]+a[1,0])}", 
        f"Average cm:", 
        f"||True 0| True 1|\n|-|-|-|\n|Predicted 0|{a[0][0]}|{a[0][1]}\n|Predicted 1|{a[1][0]}|{a[1][1]}\n", 
        f"|Acc|Spec|Loss|\n{ave}|{spec}|{loss}",'']))

def crunch_predictions(y_pred, y_true, results, conf = 0.5):
    results['y_pred'] = np.concatenate([results['y_pred'], np.squeeze(y_pred, -1)])
    results['y_true'] = np.concatenate([results['y_true'], np.squeeze(y_true, -1)])
    cm = confusionmat(y_pred, y_true, conf)
    results['acc'].append((cm[0][0] + cm[1][1])/cm.sum())
    results['spec'].append(cm[1][1]/(cm[:,1].sum()))
    results['cm'].append(cm)
    sq = np.squeeze(y_true, -1).astype('int32')
    pred = np.squeeze(y_pred, -1)
    results['loglikelihood'].append(np.log(pred[sq == 0]).sum() + np.log(1 - pred[sq == 1]).sum())