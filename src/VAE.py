import tensorflow as tf
from keras import backend as K
from itertools import product
from functools import partial
import gc

class Encoder(tf.keras.layers.Layer):
    def __init__(self, layersizes, latentsize):
        super(Encoder, self).__init__()

        self.mlp = [tf.keras.layers.Dense(layer,activation = 'relu') for layer in layersizes]
        self.means = tf.keras.layers.Dense(latentsize)
        self.logvar = tf.keras.layers.Dense(latentsize)

    def call(self, inp):
        x = inp
        for layer in self.mlp:
            x = layer(x)
        
        return self.means(x), self.logvar(x)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, layersizes, outputsize):
        super(Decoder, self).__init__()

        self.mlp = [tf.keras.layers.Dense(layer,activation = 'relu', name = f'Decoder{layer}') for layer in layersizes]


    def call(self, inp):
        x = inp
        for layer in self.mlp:
            x = layer(x)
        
        return x

class VAE(tf.keras.Model):

    def __init__(self, inputsize, inlayersize, latentsize, outlayersize = None, outputsize = None, finalactivation = ['relu']):
        """
        Parameters
        ----------
        inputsize : list
            The sizes of the layers for each input before they enter the encoder/decoder. Can be left empty
        inlayersize : list
            The sizes of each layer of the encoder
        latentsize : int
            The size of the latent layer
        outlayersize : list
            The sizes of each layer of the decoder
        outputsize : list
            Size of layers of each output, determines final size of model output. Can be left empty
        """
        super(VAE, self).__init__()
        
        if outlayersize == None:
            outlayersize = list(reversed(inlayersize))

        if outputsize == None:
            outputsize = list(reversed(inputsize))

        self.inlayers = []
        self.outlayers = []

        for inputlayer in inputsize:
            self.inlayers.append([tf.keras.layers.Dense(size, activation = 'relu') for size in inputlayer])

        for outputlayer in outputsize[:-1]:
            self.outlayers.append([tf.keras.layers.Dense(size) for size in outputlayer])
        
        if len(outputsize) > 0:
            self.outlayers.append([tf.keras.layers.Dense(size, activation = act) for size, act in zip(outputsize[-1], finalactivation)])

        self.encoder = Encoder(inlayersize, latentsize)
        self.decoder = Decoder(outlayersize, outputsize)

        self.inputsize = inputsize
        self.inputsize = inputsize
        self.inlayersize = inlayersize
        self.latentsize = latentsize
        self.outlayersize = outlayersize
        self.outputsize = outputsize
        self.finalactivation = finalactivation
        self.compile_fn = None

    def call(self, inp):
        for level in self.inlayers:
            inp = [layer(i) for layer, i in zip(level, inp)]
            
        inp = tf.concat(inp, -1)
        means, logvar = self.encoder(inp)
        
        var = tf.exp(0.5*logvar)
        
        norm = tf.random.normal([1], means, var)

        output = self.decoder(norm)

        if len(self.outlayers) > 0:
            output = [layer(output) for layer in self.outlayers[0]]
            for level in self.outlayers[1:]: # TODO: Get rid of for loops
                output = [layer(i) for layer, i in zip(level, output)]
        return output
    
    def reset_model(self):
        tf.keras.backend.clear_session()
        for l in self.inlayers:
            for i in l:
                del i
            del l
        for l in self.outlayers:
            for i in l:
                del i
            del l

        del self.encoder
        del self.decoder
        tf.keras.backend.clear_session()
        gc.collect()
        
        self.inlayers = []
        self.outlayers = []

        for inputlayer in self.inputsize:
            self.inlayers.append([tf.keras.layers.Dense(size, activation = 'relu') for size in inputlayer])

        for outputlayer in self.outputsize[:-1]:
            self.outlayers.append([tf.keras.layers.Dense(size) for size in outputlayer])
        
        if len(self.outputsize) > 0:
            self.outlayers.append([tf.keras.layers.Dense(size, activation = act) for size, act in zip(self.outputsize[-1], self.finalactivation)])

        self.encoder = Encoder(self.inlayersize, self.latentsize)
        self.decoder = Decoder(self.outlayersize, self.outputsize)
        if self.compile_fn is not None: self._compile()

    def addcompile(self, fn):
        self.compile_fn = fn

    def _compile(self):
        self.compile_fn(self)

def testloss(labels, predictions):
    e1 = tf.keras.losses.mean_squared_error(labels[0], predictions[0])
    e2 = tf.keras.losses.mean_squared_error(labels[1], predictions[1])
    return tf.math.reduce_mean([e1,e2])



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from train import train, splitdata, crossvalidation, modeltests
    import matplotlib
    import random
    import multiprocessing
    from src.tests.vaetest import *

    random.seed(0)
    np.random.seed(0)

    X,Y = testdata()

    
    Xtest, Ytest = testdata(n4=100)
    
    """ a1 = {'inputsize':[], 'inlayersize':[128,64,16], 'latentsize':8, 'outlayersize' : None, 'outputsize' : [[2,1]], 'finalactivation' : [None, 'sigmoid']}
    a2 = {'inp':X, 'labels':[X,Y], 'testdata':[Xtest,Ytest], 'name':'01', 'description' : '128|64|16, Latent of 8, 500 epochs against 100 from each distribution and 10 outliers'}
    mp = multiprocessing.Process(target = mptraining, args = [a1,a2])
    mp.start()
    mp.join() """
    """ mp = multiprocessing.Process(target = testp1, args = [X,Y,Xtest,Ytest])
    mp.start()
    mp.join() """

    X,Y = testdata(200,200,200,20)

    """ mp = multiprocessing.Process(target = testp2, args = [X,Y,Xtest,Ytest])
    mp.start()
    mp.join() """

    testp4(X,Y,Xtest,Ytest)
    """ mp = multiprocessing.Process(target = testp3, args = [X,Y,Xtest,Ytest])
    mp.start()
    mp.join() """