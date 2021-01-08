from src.CVHelper import CVHelper, crunch_predictions, writeResults
from src.data_exploration.seismicdata import rawdf, data_preprocessing
from src.train import weightedce
from functools import partial
import tensorflow as tf
import pandas as pd
import numpy as np
from imblearn.combine import SMOTETomek
from src.VAE import CustomModel, VAE

def callback(wdir):
    plateau = tf.keras.callbacks.ReduceLROnPlateau(verbose=1)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=wdir)

    earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_output_12_loss', min_delta=10e-7, patience=18, verbose=2,
    mode='auto', baseline=None, restore_best_weights=True
    )

    return [plateau, tb_callback, earlystop]

def train_preprocessing(trainin, trainout):
    preprocessing = SMOTETomek(n_jobs=-1)
    X,y = preprocessing.fit_resample(trainin, trainout.iloc[:,-1].astype('int32'))
    X['class'] = y

    inp, tar = data_preprocessing(X)

    return inp, tar

def test_preprocessing(testin, testout):
    return data_preprocessing(testout)

def data():
    raw = rawdf()
    return raw.iloc[:,:-1], raw

class CompileHelper:
    def __init__(self, losses, loss_weights):
        self.losses = losses
        self.loss_weights = loss_weights
    
    def __call__(self, model):
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer, loss=self.losses, loss_weights=self.loss_weights)
    
    def __eq__(self, other):
        return self.losses == other.losses and self.loss_weights == other.loss_weights

class WCEHelper:
    def __init__(self, weights, conf = 0.5):
        self.weights = tf.constant(weights, dtype = tf.float32)
        self.conf = conf
    
    def __call__(self, y_true, y_pred):
        return weightedce(y_true, y_pred, self.weights, self.conf)
    
    def __eq__(self, other):
        return tf.math.reduce_all(self.weights == other.weights) and self.conf == other.conf

class DoubleVAE(CustomModel):
    def __init__(self, **kwargs):
        self.m1 = VAE(**kwargs)
        self.m2 = VAE(**kwargs)

    def fit(self, **kwargs):
        trainin, trainout = kwargs['x'], kwargs['y']
        k1, k2 = {}, {}
        k1.update(kwargs)
        k2.update(kwargs)
        k1['x'] = [_[0::2]for _ in trainin]
        k2['x'] = [_[1::2]for _ in trainin]
        k1['y'] = [_[0::2]for _ in trainout]
        k2['y'] = [_[1::2]for _ in trainout]
        self.m1.fit(**k1)
        self.m2.fit(**k2)

    def reset_model(self):
        self.m1.reset_model()
        self.m2.reset_model()
    
    def predict(self, testin):
        return self.call(testin)

    def info(self):
        i1, i2 = self.m1.info, self.m2.info
        res = {}
        for key in i1:
            res[key] = [i1[key], i2[key]]
        return res
    
    def addcompile(self, fn):
        self.m1.addcompile(fn)
        self.m2.addcompile(fn)
    
    def _compile(self):
        self.m1._compile()
        self.m2._compile()
    
    def call(self, inp):
        pred1 = self.m1.predict(inp)
        pred2 = self.m2.predict(inp)
        return np.mean([pred1,pred2], 0)



def test1():
    CEweights = [[1,1],[1,1]]
    losses = [tf.losses.mean_absolute_error]*11
    ce = WCEHelper(CEweights)
    losses.append(ce)
    ce.__name__ = 'weighted_ce'
    loss_weights = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 10]
    comp = CompileHelper(losses, loss_weights)

    #comp = partial(compile_model, losses = losses, loss_weights = loss_weights)
    kwargs = {'callbacks': callback}
    vaeArgs = {'inputsize':[[4,4,4,4,4,4,4,8,8,8,8],[8,8,8,8,8,8,16,32,32,32,32]],
    'inlayersize': [512, 512, 1024],
    'latentsize': 128, 
    'outputsize':[[4,4,4,4,4,4,8,16,16,16,16,4], [1,1,1,1,1,1,2,4,4,4,4,1]],
    'finalactivation':[None,None,None,None,None,None,None,None,None,None,None,'sigmoid']}
    cvh = CVHelper(vaeArgs, data, comp, '01', cvRuns = 2,
                description = f'losses: {[l.__name__ for l in losses]}\nloss_weights: {loss_weights}\nCEweights: {CEweights}', 
                epochs=1, 
                k = 10, 
                seed = 0, 
                verbose = 2, 
                testsplit = (1, 20), 
                wdir = 'src/reports/test2',
                train_preprocessing = train_preprocessing,
                test_preprocessing = test_preprocessing,
                **kwargs)
                
    cvh.crossvalidation()

data2seed = 0

def testdata2(s1  = 120, s2 = 12):
    np.random.seed(data2seed)
    data2seed += 1

    center = np.random.uniform(-5, 5, [s1])
    norm = np.random.normal(center, size = [s1])
    uni = np.random.uniform(-7,7,[s2,2])
    df0 = pd.DataFrame(center, norm).T
    df0['class'] = 0
    df1 = pd.DataFrame(uni)
    df1['class'] = 1
    df0.append(df1)
    return df0.iloc[:,:-1], df0

def testresults2(file, results):
    file.write('\n'.join([f'Test result on unseen data:','']))
    writeResults(file, results)

def testpost2(model, results):
    testin, testout = testdata2(100, 100)
    testin, testout = test_preprocessing2(testin, testout)
    
    pred = model.predict(testin)

    crunch_predictions(pred[-1], testout[-1], results)

def data_preprocessing2(trainin):
    lst = [trainin[x] for x in trainin.columns]
    return lst[:-1], lst

def train_preprocessing2(trainin, trainout):
    preprocessing = SMOTETomek(n_jobs=-1)
    X,y = preprocessing.fit_resample(trainin, trainout.iloc[:,-1].astype('int32'))
    X['class'] = y

    return data_preprocessing2(trainout)


def test_preprocessing2(testin, testout):
    return data_preprocessing2(trainout)

def test2():
    CEweights = [[1,1],[1,1]]
    losses = [tf.losses.mean_absolute_error]*2
    ce = WCEHelper(CEweights)
    losses.append(ce)
    ce.__name__ = 'weighted_ce'
    loss_weights = [1,1,1]
    comp = CompileHelper(losses, loss_weights)

    kwargs = {'callbacks': callback, 'model': DoubleVAE, 
    'postresult': testresults2, 'postfold': testpost2}
    vaeArgs = {'inputsize':[[4,4,4,4,4,4,4,8,8,8,8],[8,8,8,8,8,8,16,32,32,32,32]],
    'inlayersize': [512, 512, 1024],
    'latentsize': 128, 
    'outputsize':[[4,4,4,4,4,4,8,16,16,16,16,4], [1,1,1,1,1,1,2,4,4,4,4,1]],
    'finalactivation':[None,None,None,None,None,None,None,None,None,None,None,'sigmoid']}
    cvh = CVHelper(vaeArgs, testdata2, comp, '01', cvRuns = 2,
                description = f'Model has 2 inputs, class 0 is where the second input is drawn from a normal distribution with mean of the first input, class 1 is where it isnt
                \nlosses: {[l.__name__ for l in losses]}\nloss_weights: {loss_weights}\nCEweights: {CEweights}', 
                epochs=1, 
                k = 10, 
                seed = 0, 
                verbose = 2, 
                testsplit = (1, 20), 
                wdir = 'src/reports/test2',
                train_preprocessing = train_preprocessing2,
                test_preprocessing = test_preprocessing2,
                **kwargs)
                
    cvh.crossvalidation()


if __name__ == "__main__":
    #test1()
    test2()