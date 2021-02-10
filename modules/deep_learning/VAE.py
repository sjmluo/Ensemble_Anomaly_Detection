import modules.deep_learning.models as models
import numpy as np
import tensorflow as tf

class Model:
    def fit(self, x):
        self.model = None
        self.model.fit(x)

    def predict(self, x):
        """
        Give predicted classes of [0,1] in shape [?]
        """
        y_pred = self.model.predict(x)
        return y_pred

    def predict_proba(self, x):
        """
        return probabilities for class and class 1 in shape [2,?]
        """
        y_pred = self.model.predict_proba(x)
        return y_pred

class CompileHelper:
    def __init__(self, losses, loss_weights):
        self.losses = losses
        self.loss_weights = loss_weights
    
    def __call__(self, model):
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        model.compile(optimizer, loss=self.losses, loss_weights=self.loss_weights, run_eagerly = False)
    
    def __eq__(self, other):
        return self.losses == other.losses and self.loss_weights == other.loss_weights

def ignore(y_true, y_pred):
    return 0.

class ReconstructionVAE(Model):
    def __init__(self):
        self._model = models.VAEdistance
        self.verbose = 0
        self.epochs = 500
    
    def fit(self, x):
        layersizes = np.array([2**w for w in range(2,12)])
        layersizes = layersizes[layersizes < x.shape[1]]

        if len(layersizes) > 5:
            layersizes = layersizes[::2]

        losses = [tf.losses.mean_absolute_error]*2
        losses.append(ignore)
        loss_weights = [1,0]
        ch = CompileHelper(losses, loss_weights)


        self.model = self._model(inputsize = [], inlayersize = layersizes,
             outputsize = [[x.shape[1]]],latentsize = 4, finalactivation = ['linear'])
            
        ch(self.model)

        y = [x, np.zeros([x.shape[0], 1])]


        self.model.fit(x, y, callbacks = self.callbacks(), 
        verbose = self.verbose, epochs = self.epochs)
    
    def callbacks(self):
        plateau = tf.keras.callbacks.ReduceLROnPlateau(verbose=1, patience=10, monitor='loss')
        earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='loss', min_delta=10e-7, patience=25, verbose=2,
        mode='auto', baseline=None, restore_best_weights=True
        )

        return [plateau, earlystop]
    
    def predict_proba(self, x):
        y_pred = self.model.predict(x)[0]
        y_pred = np.mean(y_pred, 1)
        y_pred = np.squeeze(y_pred)
        y_pred = (y_pred - y_pred.min())/(y_pred.max() - y_pred.min())
        class0 = 1-y_pred
        return [class0, y_pred]

    def predict(self, x, threshold = 'auto'):
        y_pred = self.predict_proba(x)
        y_pred = y_pred[-1]
        if threshold == 'auto':
            t_val = np.quantile(y_pred, 0.95)
        else:
            t_val = y_pred < 0.5
        return np.where(y_pred < t_val, 0, 1)
    
class VAErcp(Model):
    def __init__(self):
        self._model = models.VAErcp
        self.verbose = 0
        self.epochs = 500
    
    def fit(self, x):
        layersizes = np.array([2**w for w in range(2,12)])
        layersizes = layersizes[layersizes < x.shape[1]]

        if len(layersizes) > 5:
            layersizes = layersizes[::2]

        losses = [tf.losses.mean_absolute_error]*2
        losses.append(ignore)
        loss_weights = [1,-1]
        ch = CompileHelper(losses, loss_weights)


        self.model = self._model(inputsize = [], inlayersize = layersizes,
             outputsize = [[x.shape[1]]],latentsize = 4)
            
        ch(self.model)

        y = [x, np.zeros([x.shape[0], 1])]


        self.model.fit(x, y, callbacks = self.callbacks(), 
        verbose = self.verbose, epochs = self.epochs)
    
    def callbacks(self):
        plateau = tf.keras.callbacks.ReduceLROnPlateau(verbose=1, patience=10, monitor='loss')
        earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='loss', min_delta=10e-7, patience=25, verbose=2,
        mode='auto', baseline=None, restore_best_weights=True
        )

        return [plateau, earlystop]
    
    def predict_proba(self, x):
        y_pred = self.model.predict(x)[0]
        y_pred = np.mean(y_pred, 1)
        y_pred = np.squeeze(y_pred)
        y_pred = (y_pred - y_pred.min())/(y_pred.max() - y_pred.min())
        class0 = 1-y_pred
        return [class0, y_pred]

    def predict(self, x, threshold = 'auto'):
        y_pred = self.predict_proba(x)
        y_pred = y_pred[-1]
        if threshold == 'auto':
            t_val = np.quantile(y_pred, 0.95)
        else:
            t_val = y_pred < 0.5
        return np.where(y_pred < t_val, 0, 1)
    
class VAEvampprior(Model):
    def __init__(self):
        self._model = models.Vamprior
        self.verbose = 0
        self.epochs = 500
    
    def fit(self, x):
        print('fit')
        layersizes = np.array([2**w for w in range(2,12)])
        layersizes = layersizes[layersizes < x.shape[1]]

        if len(layersizes) > 5:
            layersizes = layersizes[::2]

        losses = [tf.losses.mean_absolute_error]*2
        losses.append(ignore)
        loss_weights = [1,5]
        ch = CompileHelper(losses, loss_weights)

        inputshape = list(x.shape)
        inputshape[0] = -1
        inputshape.insert(0,1)

        self.model = self._model(inputsize = [], inlayersize = layersizes,
             outputsize = [[x.shape[1]]],latentsize = 4, inputshape = inputshape)
            
        ch(self.model)

        y = [x, np.zeros([x.shape[0], 1])]


        self.model.fit([x], y, callbacks = self.callbacks(), 
        verbose = self.verbose, epochs = self.epochs)
    
    def callbacks(self):
        plateau = tf.keras.callbacks.ReduceLROnPlateau(verbose=1, patience=10, monitor='loss')
        earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='loss', min_delta=10e-7, patience=25, verbose=2,
        mode='auto', baseline=None, restore_best_weights=True
        )

        return [plateau, earlystop]
    
    def predict_proba(self, x):
        y_pred = self.model.predict(x)[0]
        y_pred = np.mean(y_pred, 1)
        y_pred = np.squeeze(y_pred)
        y_pred = (y_pred - y_pred.min())/(y_pred.max() - y_pred.min())
        class0 = 1-y_pred
        return [class0, y_pred]

    def predict(self, x, threshold = 'auto'):
        y_pred = self.predict_proba(x)
        y_pred = y_pred[-1]
        if threshold == 'auto':
            t_val = np.quantile(y_pred, 0.95)
        else:
            t_val = y_pred < 0.5
        return np.where(y_pred < t_val, 0, 1)
    