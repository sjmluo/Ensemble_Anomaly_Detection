from src.CVHelper import CVHelper
from src.data_exploration.seismicdata import rawdf, data_preprocessing
from src.train import weightedce
from functools import partial
import tensorflow as tf
from imblearn.combine import SMOTETomek

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

class WCEHelper:
    def __init__(self, weights, conf = 0.5):
        self.weights = tf.constant(weights, dtype = tf.float32)
        self.conf = conf
    
    def __call__(self, y_true, y_pred):
        return weightedce(y_true, y_pred, self.weights, self.conf)


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
                description = f'losses: {losses}\nloss_weights: {loss_weights}\nCEweights: {CEweights}', 
                epochs=10, 
                k = 10, 
                seed = 0, 
                verbose = 2, 
                testsplit = (1, 20), 
                wdir = 'src/reports/test2',
                train_preprocessing = train_preprocessing,
                test_preprocessing = test_preprocessing,
                **kwargs)
                
    cvh.crossvalidation()

if __name__ == "__main__":
    test1()