from src.CVHelper import CVHelper, crunch_predictions, writeResults
from src.data_exploration.seismicdata import rawdf, data_preprocessing
from src.train import weightedce, confusionmat, bestalpha
from functools import partial
import tensorflow as tf
import pandas as pd
import numpy as np
from itertools import zip_longest
from imblearn.combine import SMOTETomek
from src.VAE import CustomModel, VAE, SVAE, VAErcp, VAErcp2, VAEdistance

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
        super(CustomModel, self).__init__()

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
        h1 = self.m1.fit(**k1)
        h2 = self.m2.fit(**k2)
        hist = tf.keras.callbacks.History()
        hist.epoch = np.array(list(zip_longest(h2.epoch, h1.epoch)), dtype = 'float32').astype('int32')
        for key in h1.history:
            hist.history[key] = np.array(list(zip_longest(h2.history[key], h1.history[key])), 'float32')
        return hist

    def reset_model(self):
        self.m1.reset_model()
        self.m2.reset_model()
    
    def predict(self, testin):
        return self.call(testin)

    def info(self):
        i1, i2 = self.m1.info(), self.m2.info()
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
                wdir = 'src/reports/test2',
                train_preprocessing = train_preprocessing,
                test_preprocessing = test_preprocessing,
                **kwargs)
                
    cvh.crossvalidation()

def testdata2(s1  = 120, s2 = 12, seed = 0):
    np.random.seed(seed)

    center = np.random.uniform(-5, 5, [s1])
    norm = np.random.normal(center, size = [s1])
    uni = np.random.uniform(-7,7,[s2,2])
    df0 = pd.DataFrame([center, norm]).T
    df0['class'] = 0
    df1 = pd.DataFrame(uni)
    df1['class'] = 1
    df0 = df0.append(df1)
    df0['class'] = df0['class'].astype('int32')
    return df0.iloc[:,:-1], df0

def testresults2(file, results):
    file.write('\n'.join([f'Test result on unseen data:','']))
    writeResults(file, results)

def testpost2(model, results):
    if results == {}:
        results.update({'acc':[], 'loss':[], 'cm':[], 'spec':[], 'y_pred': [], 'y_true': [], 'loglikelihood': []})
    testin, testout = testdata2(100, 100, seed = 40)
    testin, testout = test_preprocessing2(testin, testout)
    
    pred = model.predict(testin)

    crunch_predictions(pred[-1], testout[-1], results)

def data_preprocessing2(trainin):
    lst = [np.expand_dims(trainin[x].to_numpy(),-1) for x in trainin.columns]
    return lst[:-1], lst

def train_preprocessing2(trainin, trainout):
    preprocessing = SMOTETomek(n_jobs=-1)
    X,y = preprocessing.fit_resample(trainin, trainout.iloc[:,-1].astype('int32'))
    X['class'] = y

    return data_preprocessing2(trainout)


def test_preprocessing2(testin, testout):
    return data_preprocessing2(testout)

def callback2(wdir):
    plateau = tf.keras.callbacks.ReduceLROnPlateau(verbose=1)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=wdir)

    earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_output_3_loss', min_delta=10e-7, patience=18, verbose=2,
    mode='auto', baseline=None, restore_best_weights=True
    )

    return [plateau, tb_callback, earlystop]

def test2():
    CEweights = [[1,10],[1,1]]
    losses = [tf.losses.mean_absolute_error]*2
    ce = WCEHelper(CEweights)
    losses.append(ce)
    ce.__name__ = 'weighted_ce'
    loss_weights = [1,1,1]
    comp = CompileHelper(losses, loss_weights)

    kwargs = {'callbacks': callback2, 'model': DoubleVAE, 
    'postresult': testresults2, 'postfold': testpost2}
    vaeArgs = {'inputsize':[[4,4],[8,8]],
    'inlayersize': [64, 32, 16],
    'latentsize': 2, 
    'outputsize':[[4,4,4], [1,1,1]],
    'finalactivation':[None,None,'sigmoid']}
    cvh = CVHelper(vaeArgs, testdata2, comp, '08', cvRuns = 5,
                description = f'Model has 2 inputs, class 0 is where the second input is drawn from a normal distribution with mean of the first input, class 1 is where it isnt. Double training data \
                \nlosses: {[l.__name__ for l in losses]}\nloss_weights: {loss_weights}\nCEweights: {CEweights}', 
                epochs=500, 
                k = 10, 
                seed = 0, 
                verbose = 1, 
                wdir = 'src/reports/test2',
                train_preprocessing = train_preprocessing2,
                test_preprocessing = test_preprocessing2,
                **kwargs)
                
    cvh.crossvalidation()

def test3():
    CEweights = [[1,1],[1,1]]
    losses = [tf.losses.mean_absolute_error]*2
    ce = WCEHelper(CEweights)
    losses.append(ce)
    ce.__name__ = 'weighted_ce'
    loss_weights = [1,1,1]
    comp = CompileHelper(losses, loss_weights)

    kwargs = {'callbacks': callback2, 'model': SVAE, 
    'postresult': testresults2, 'postfold': testpost2}
    vaeArgs = {'inputsize':[[4,4],[8,8]],
    'inlayersize': [64, 32, 16],
    'latentsize': 2, 
    'outputsize':[[4,4], [1,1]],
    'finalactivation':[None,None,'sigmoid'],
    'fc_size': [64,32,1]}
    {'encoder_input_size', 'fc_size', 'decoder_output_size', 'decoder_activation', 'latent_size'}

    cvh = CVHelper(vaeArgs, testdata2, comp, '06', cvRuns = 5,
                description = f'Model has 2 inputs, class 0 is where the second input is drawn from a normal distribution with mean of the first input, class 1 is where it isnt. Double training data \
                \nlosses: {[l.__name__ for l in losses]}\nloss_weights: {loss_weights}\nCEweights: {CEweights}', 
                epochs=500, 
                k = 10, 
                seed = 0, 
                verbose = 1, 
                wdir = 'src/reports/test3',
                train_preprocessing = train_preprocessing2,
                test_preprocessing = test_preprocessing2,
                **kwargs)
                
    cvh.crossvalidation()

class KLD:
    def __init__(self, latent_size):
        self.latent_size = latent_size
        self.__name__ = 'KLD'
    
    def __call__(self, y_true, y_pred):
        logvar = y_pred[:,:self.latent_size]
        means = y_pred[:,self.latent_size:]
        return -0.5*tf.math.reduce_mean(1 + logvar - tf.square(means) - tf.exp(logvar))
    
    def __eq__(self, other):
        return self.latent_size == other.latent_size

def callback4(wdir):
    plateau = tf.keras.callbacks.ReduceLROnPlateau(verbose=1)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=wdir)

    earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_output_4_loss', min_delta=10e-7, patience=18, verbose=2,
    mode='auto', baseline=None, restore_best_weights=True
    )

    return [plateau, tb_callback, earlystop]

def test4():
    latent_size = 16
    CEweights = [[1,10],[1,1]]
    losses = [tf.losses.mean_absolute_error]*2
    ce = WCEHelper(CEweights)
    kld = KLD(latent_size)
    losses.append(kld)
    losses.append(ce)
    ce.__name__ = 'weighted_ce'
    loss_weights = [1,1,1,10]
    comp = CompileHelper(losses, loss_weights)
    

    kwargs = {'callbacks': callback4, 'model': SVAE, 
    'postresult': testresults2, 'postfold': testpost2}
    vaeArgs = {'inputsize':[[4,4],[8,8]],
    'inlayersize': [128, 512, 512],
    'latentsize': latent_size, 
    'outputsize':[[4,4], [1,1]],
    'finalactivation':[None,None,'sigmoid'],
    'fc_size': [64,32,1]}
    {'encoder_input_size', 'fc_size', 'decoder_output_size', 'decoder_activation', 'latent_size'}

    cvh = CVHelper(vaeArgs, testdata2, comp, '15', cvRuns = 5,
                description = f'Model has 2 inputs, class 0 is where the second input is drawn from a normal distribution\
with mean of the first input, class 1 is where it isnt. Proper implementation of SVAE\
\nlosses: {[l.__name__ for l in losses]}\nloss_weights: {loss_weights}\nCEweights: {CEweights}', 
                epochs=500, 
                k = 10, 
                seed = 0, 
                verbose = 2, 
                wdir = 'src/reports/test3',
                train_preprocessing = train_preprocessing4,
                test_preprocessing = test_preprocessing4,
                **kwargs)
                
    cvh.crossvalidation()

def data_preprocessing4(trainin):
    lst = [np.expand_dims(trainin[x].to_numpy(),-1) for x in trainin.columns]
    lst.insert(2, np.zeros([len(lst[0]),8]))
    return lst[:2], lst

def train_preprocessing4(trainin, trainout):
    preprocessing = SMOTETomek(n_jobs=-1)
    X,y = preprocessing.fit_resample(trainin, trainout.iloc[:,-1].astype('int32'))
    X['class'] = y

    return data_preprocessing4(trainout)


def test_preprocessing4(testin, testout):
    return data_preprocessing4(testout)

class customloss5:
    def __init__(self, latent_size):
        self.latent_size = latent_size
    
    def __call__(self, y_true, y_pred):
        mu = y_pred[:,:self.latent_size]
        #logvar = y_pred[self.latent_size:self.latent_size*2]
        var = y_pred[:,self.latent_size*2:self.latent_size*3]
        mu_post = y_pred[:,self.latent_size*3:self.latent_size*4]
        var_post = y_pred[:,self.latent_size*4:self.latent_size*5]
        return tf.keras.losses.MSE(mu,mu_post) + tf.keras.losses.MSE(var, var_post)

def callback5(wdir):
    plateau = tf.keras.callbacks.ReduceLROnPlateau(verbose=1)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=wdir)

    earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=10e-7, patience=18, verbose=2,
    mode='auto', baseline=None, restore_best_weights=True
    )

    return [plateau, tb_callback, earlystop]

def ignore(y_true, y_pred):
    return 0

def testpost5(model, results):
    if results == {}:
        results.update({'acc':[], 'loss':[], 'cm':[], 'spec':[], 'y_pred': [], 'y_true': [], 'loglikelihood': [], 'alpha': []})
    testin, testout = testdata2(100, 100, seed = 40)
    testin, testout = test_preprocessing2(testin, testout)
    
    pred = model.predict(testin)

    highestalpha, highestf1 = bestalpha(testout[-1], pred[-1])
    results['alpha'].append(highestalpha)

    crunch_predictions(pred[-1], testout[-1], results, highestalpha)

def test5():
    latent_size = 4
    cl5 = customloss5(latent_size)
    cl5.__name__ = 'cl5'
    CEweights = [[1,10],[1,1]]
    losses = [tf.losses.mean_absolute_error]*2
    losses.insert(0, cl5)
    losses.append(ignore)
    loss_weights = [1,1,1,0] #meanvar, point1, point2, norm/epoch
    comp = CompileHelper(losses, loss_weights)
    

    kwargs = {'callbacks': callback5, 'model': VAErcp, 
    'postresult': testresults2, 'postfold': testpost5}
    vaeArgs = {'inputsize':[[4,4],[8,8]],
    'inlayersize': [32, 64, 128],
    'latentsize': latent_size, 
    'outputsize':[[1,1]],
    'finalactivation':[None,None,'sigmoid']}

    cvh = CVHelper(vaeArgs, testdata2, comp, '01', cvRuns = 5,
                description = f'Model has 2 inputs, class 0 is where the second input is drawn from a normal distribution\
with mean of the first input, class 1 is where it isnt. Proper implementation of SVAE\
\nlosses: {[l.__name__ for l in losses]}\nloss_weights: {loss_weights}\nCEweights: {CEweights}', 
                epochs=500, 
                k = 10, 
                seed = 0, 
                verbose = 2, 
                wdir = 'src/reports/test4',
                train_preprocessing = train_preprocessing5,
                test_preprocessing = test_preprocessing5,
                **kwargs)
                
    cvh.crossvalidation()

def data_preprocessing5(trainin):
    lst = [np.expand_dims(trainin[x].to_numpy(),-1) for x in trainin.columns]
    lst.insert(0, np.zeros([len(lst[0]),8]))
    return lst[1:3], lst

def train_preprocessing5(trainin, trainout):
    preprocessing = SMOTETomek(n_jobs=-1)
    X,y = preprocessing.fit_resample(trainin, trainout.iloc[:,-1].astype('int32'))
    X['class'] = y

    return data_preprocessing5(trainout)

def test_preprocessing5(testin, testout):
    return data_preprocessing5(testout)


def testpost6(model, results):
    if results == {}:
        results.update({'acc':[], 'loss':[], 'cm':[], 'spec':[], 'y_pred': [], 'y_true': [], 'loglikelihood': [], 'alpha': [], 'y_inp': []})
    testin, testout = testdata2(100, 100, seed = 40)
    testin, testout = test_preprocessing2(testin, testout)
    if results['y_inp'] == []:
        results['y_inp'] = np.squeeze(testin,-1).T
    else:
        results['y_inp'] = np.concatenate([results['y_inp'], np.squeeze(testin,-1).T])
    
    pred = model.predict(testin)

    highestalpha, highestf1 = bestalpha(testout[-1], pred[-1])
    results['alpha'].append(highestalpha)

    crunch_predictions(pred[-1], testout[-1], results, highestalpha)

def test6():
    CEweights = [[1,1],[1,1]]
    losses = [tf.losses.mean_absolute_error]*2
    losses.append(ignore)
    loss_weights = [1,1,0]
    comp = CompileHelper(losses, loss_weights)

    kwargs = {'callbacks': callback5, 'model': VAEdistance, 
    'postresult': testresults2, 'postfold': testpost6}
    vaeArgs = {'inputsize':[[4,4],[8,8]],
    'inlayersize': [64, 32, 16],
    'latentsize': 4, 
    'outputsize':[[4,4], [1,1]],
    'finalactivation':[None,None,None]}
    {'encoder_input_size', 'fc_size', 'decoder_output_size', 'decoder_activation', 'latent_size'}

    cvh = CVHelper(vaeArgs, testdata2, comp, '05', cvRuns = 5,
                description = f'Model has 2 inputs, class 0 is where the second input is drawn from a normal distribution with mean of the first input, class 1 is where it isnt. using distance as outlier metric\
                \nlosses: {[l.__name__ for l in losses]}\nloss_weights: {loss_weights}\nCEweights: {CEweights}', 
                epochs=500, 
                k = 10, 
                seed = 0, 
                verbose = 1, 
                wdir = 'src/reports/test4',
                train_preprocessing = train_preprocessing2,
                test_preprocessing = test_preprocessing2,
                **kwargs)
                
    cvh.crossvalidation()

if __name__ == "__main__":
    #test1()
    #test2()

    #test3()
    #test4()
    #test5()
    test6()
    """ vaeArgs = {'inputsize':[[4,4],[8,8]],
    'inlayersize': [64, 32, 16],
    'latentsize': 2, 
    'outputsize':[[4,4], [1,1]],
    'finalactivation':[None,None,'sigmoid'],
    'fc_size': [64,32,1]}
    {'encoder_input_size', 'fc_size', 'decoder_output_size', 'decoder_activation', 'latent_size'}
    svae = SVAE(**vaeArgs)
    inp, tar = testdata2()
    inp = inp.to_numpy()
    tar = tar.to_numpy()

    tar = [np.expand_dims(tar[:,x],-1) for x in range(3)]
    inp = [np.expand_dims(inp[:,x],-1) for x in range(2)]

    CEweights = [[1,1],[1,1]]
    loss = SVAEloss()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    svae.compile(optimizer, loss= loss, run_eagerly=True)
    svae.fit(inp, tar) """
    