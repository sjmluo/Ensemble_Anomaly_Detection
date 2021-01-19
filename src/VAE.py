import tensorflow as tf
from keras import backend as K
from itertools import product
from functools import partial
import gc
import tensorflow_probability as tfp
import numpy as np

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

class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
    
    def info(self) -> dict:
        """
        Provides descriptive information on model (structure, loss, etc.)
        """
        pass

    def reset_model(self) -> None:
        """
        Resets model such that all weights and reset
        """
        pass

    def addcompile(self, fn) -> None:
        """
        Adds a compile function that compiles model each time it is reset or when a save is loaded. Model function should have signature compile_fn(model)
        """
        self.compile_fn = fn

    def _compile(self):
        """
        Compiles model
        """
        self.compile_fn(self)


class VAE(CustomModel):

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
            self.outlayers.append([tf.keras.layers.Dense(size, activation = 'relu') for size in outputlayer])
        
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

    def info(self):
        VAEargs = ['inputsize', 'inlayersize', 'latentsize', 'outlayersize', 'outputsize', 'finalactivation']
        res = {}
        for arg in VAEargs:
            res[arg] = self.__dict__[arg]
        return res

class SVAE(CustomModel):
    """
    A similar implementation of https://arxiv.org/abs/2012.08637
    https://github.com/tianchenji/Multimodal-SVAE
    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        inputsize : list
            The sizes of the layers for each input before they enter the encoder/decoder. Default [16, 32, 64]
        inlayersize : list
            A 2D list specifying the sizes of each layer of the encoder, eg. [[4,4],[8,8]] means 2 inputs that are passed into Dense layer of size 4
             then uncombined outputs enter next layer of size 8 Default []
        latentsize : int
            The size of the latent layer. Default 4
        outlayersize : list
            The sizes of each layer of the decoder, post processing after decoder. Default reverse of inlayersize
        outputsize : list
            Size of layers of each output, determines final size of model output. Default reverse of inputsize
        fc_size: list
            List of int, size of each layer of fully connected layers
        finalactivation:
            List of activations to use for the final output layer. Should be same size as last outputlayersize
        No checking of args are in place#TODO
        """
        super(SVAE, self).__init__()
        self.inputsize = [16,32,64]
        self.inlayersize = []
        self.latentsize = 4
        self.fc_size = [64, 32, 1]
        allowed_kwargs = {'inputsize', 'inlayersize', 'latentsize', 'outlayersize', 'outputsize', 'fc_size', 'finalactivation'}

        for k in kwargs:
            if k not in allowed_kwargs:
                raise ValueError(f'{k} is not a valid kwarg')
        
        self.__dict__.update(kwargs)
        
        if 'outlayersize' not in kwargs:
            self.outlayersize = list(reversed(self.inlayersize))

        if 'outputsize' not in kwargs:
            self.outputsize = list(reversed(self.inputsize))

        self.inlayers = []
        self.outlayers = []

        for inputlayer in self.inputsize:
            self.inlayers.append([tf.keras.layers.Dense(size, activation = 'relu') for size in inputlayer])

        for outputlayer in self.outputsize[:-1]:
            self.outlayers.append([tf.keras.layers.Dense(size, activation = 'relu') for size in outputlayer])

        if len(self.outputsize) > 0:
            self.outlayers.append([tf.keras.layers.Dense(size, activation = act) for size, act in zip(self.outputsize[-1], self.finalactivation)])

        self.encoder = Encoder(self.inlayersize, self.latentsize)
        self.fc_layers = [tf.keras.layers.Dense(s, activation = 'relu') for s in self.fc_size[:-1]]
        self.fc_layers.append(tf.keras.layers.Dense(self.fc_size[-1], activation = 'sigmoid'))
        self.decoder = Decoder(self.outlayersize, self.outputsize)


    def call(self, inp):
        for level in self.inlayers:
            inp = [layer(i) for layer, i in zip(level, inp)]
        
        inp = tf.concat(inp, -1)
        (means, logvar) = self.encoder(inp)
        encout = tf.concat([means, logvar], -1)

        var = tf.exp(0.5*logvar)
        
        norm = tf.random.normal([1], means, var)

        output = self.decoder(norm)

        if len(self.outlayers) > 0:
            output = [layer(output) for layer in self.outlayers[0]]
            for level in self.outlayers[1:]: # TODO: Get rid of for loops
                output = [layer(i) for layer, i in zip(level, output)]

        for layer in self.fc_layers:
            encout = layer(encout)
        output.append(tf.concat([logvar, means],-1))
        output.append(encout)

        return output
        
    def reset_model(self):
        tf.keras.backend.clear_session()
        self.encoder = Encoder(self.inlayersize, self.latentsize)
        self.fc_layers = [tf.keras.layers.Dense(s, activation = 'relu') for s in self.fc_size[:-1]]
        self.fc_layers.append(tf.keras.layers.Dense(self.fc_size[-1], activation = 'sigmoid'))
        self.decoder = Decoder(self.outlayersize, self.outputsize)

        if self.compile_fn is not None: self._compile()

    def addcompile(self, fn):
        self.compile_fn = fn

    def _compile(self):
        self.compile_fn(self)

    def info(self):
        VAEargs = ['inputsize', 'inlayersize', 'latentsize', 'outlayersize', 'outputsize', 'fc_size']
        res = {}
        for arg in VAEargs:
            res[arg] = self.__dict__[arg]
        return res
        
class VAErcp(CustomModel):
    """
    VAE with reconstruction probability
    https://www.semanticscholar.org/paper/Variational-Autoencoder-based-Anomaly-Detection-An-Cho/061146b1d7938d7a8dae70e3531a00fceb3c78e8
    https://arxiv.org/abs/1802.03903
    https://github.com/NetManAIOps/donut
    """

    def __init__(self, inputsize, inlayersize, latentsize, outlayersize = None, outputsize = None):
        """
        Currently only one level of post processing after Decoder is allowed, predicting of mean and variance is added automatically to output
        Parameters
        ----------
        inputsize : list
            The sizes of the layers for each input before they enter the encoder/decoder. Can be left empty
        inlayersize : list
            A 2D list specifying the sizes of each layer of the encoder, eg. [[4,4],[8,8]] means 2 inputs that are passed into Dense layer of size 4
             then uncombined outputs enter next layer of size 8
        latentsize : int
            The size of the latent layer
        outlayersize : list
            The sizes of each layer of the decoder
        outputsize : list
            Size of layers of each output, determines final size of model output. Can be left empty
        """
        super(VAErcp, self).__init__()


        if outlayersize == None:
            outlayersize = list(reversed(inlayersize))

        if outputsize == None:
            outputsize = list(reversed(inputsize))


        self.inlayers = []
        self.outlayers = []
        outputsize[-1].insert(0, latentsize)
        outputsize[-1].insert(1, latentsize)

        for inputlayer in inputsize:
            self.inlayers.append([tf.keras.layers.Dense(size, activation = 'relu') for size in inputlayer])

        for outputlayer in outputsize[:-1]:
            self.outlayers.append([tf.keras.layers.Dense(size, activation = 'relu') for size in outputlayer])
        
        if len(outputsize) > 0:
            self.outlayers.append([tf.keras.layers.Dense(size) for size in outputsize[-1]])

        self.encoder = Encoder(inlayersize, latentsize)
        self.decoder = Decoder(outlayersize, outputsize)

        self.inputsize = inputsize
        self.inputsize = inputsize
        self.inlayersize = inlayersize
        self.latentsize = latentsize
        self.outlayersize = outlayersize
        self.outputsize = outputsize
        self.compile_fn = None

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
            self.outlayers.append([tf.keras.layers.Dense(size, activation = 'relu') for size in outputlayer])
        
        if len(self.outputsize) > 0:
            self.outlayers.append([tf.keras.layers.Dense(size) for size in self.outputsize[-1]])

        self.encoder = Encoder(self.inlayersize, self.latentsize)
        self.decoder = Decoder(self.outlayersize, self.outputsize)
        if self.compile_fn is not None: self._compile()

    def call(self, inp):

        for level in self.inlayers:
            inp = [layer(i) for layer, i in zip(level, inp)]
            
        inp = tf.concat(inp, -1)
        means, logvar = self.encoder(inp)
        
        var = tf.exp(0.5*logvar)
        
        norm = tf.random.normal([1], means, var)

        output = self.call_decoder(norm)
        out = [tf.concat([means, logvar, var, output[0], output[1]], axis = 1),  *output[2:], norm]
        return out

    def call_decoder(self, norm):
        output = self.decoder(norm)

        if len(self.outlayers) > 0:
            output = [layer(output) for layer in self.outlayers[0]]
            for level in self.outlayers[1:]: # TODO: Get rid of for loops
                output = [layer(i) for layer, i in zip(level, output)]
        return output


    def predict(self, inp, L = 50):
        """
        Reconstruction probability
        """
        probs = np.zeros([len(inp)])
        out = self.call(inp)
        y_pred = out[0]
        norm = out[3]

        mu_post = y_pred[:, self.latentsize*3:self.latentsize*4]
        var_post = y_pred[:, self.latentsize*4:self.latentsize*5]

        normal = tfp.distributions.Normal(mu_post, var_post)
        probs = normal.prob(norm)

        for l in range(1, L):
            out = self.call_decoder(norm)
            mu_post = out[0]
            var_post = out[1]

            normal = tfp.distributions.Normal(mu_post, var_post)
            probs += normal.prob(norm)
        probs = np.nanmean(probs, axis = 1)
        return [np.expand_dims( 1 - (probs/L), -1)]


    def info(self):
        VAEargs = ['inputsize', 'inlayersize', 'latentsize', 'outlayersize', 'outputsize']
        res = {}
        for arg in VAEargs:
            res[arg] = self.__dict__[arg]
        return res

    def addcompile(self, fn):
        self.compile_fn = fn

    def _compile(self):
        self.compile_fn(self)

class VAErcp2(CustomModel):
    """
    Another implementation using reconstruction probability by passing through output through encoder
    """
    def __init__(self, inputsize, inlayersize, latentsize, outlayersize = None, outputsize = None):
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
        super(VAErcp2, self).__init__()


        if outlayersize == None:
            outlayersize = list(reversed(inlayersize))

        if outputsize == None:
            outputsize = list(reversed(inputsize))


        self.inlayers = []
        self.outlayers = []

        for inputlayer in inputsize:
            self.inlayers.append([tf.keras.layers.Dense(size, activation = 'relu') for size in inputlayer])

        for outputlayer in outputsize[:-1]:
            self.outlayers.append([tf.keras.layers.Dense(size, activation = 'relu') for size in outputlayer])
        
        if len(outputsize) > 0:
            self.outlayers.append([tf.keras.layers.Dense(size) for size in outputsize[-1]])

        self.encoder = Encoder(inlayersize, latentsize)
        self.decoder = Decoder(outlayersize, outputsize)

        self.inputsize = inputsize
        self.inputsize = inputsize
        self.inlayersize = inlayersize
        self.latentsize = latentsize
        self.outlayersize = outlayersize
        self.outputsize = outputsize
        self.compile_fn = None

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
            self.outlayers.append([tf.keras.layers.Dense(size, activation = 'relu') for size in outputlayer])
        
        if len(self.outputsize) > 0:
            self.outlayers.append([tf.keras.layers.Dense(size) for size in self.outputsize[-1]])

        self.encoder = Encoder(self.inlayersize, self.latentsize)
        self.decoder = Decoder(self.outlayersize, self.outputsize)
        if self.compile_fn is not None: self._compile()

    def callEncZ(self, inp):
        for level in self.inlayers:
            inp = [layer(i) for layer, i in zip(level, inp)]
            
        inp = tf.concat(inp, -1)
        means, logvar = self.encoder(inp)
        
        var = tf.exp(0.5*logvar)
        
        norm = tf.random.normal([1], means, var)

        return means, logvar, var, norm


    def call(self, inp):
        #meanvar, point1, point2, norm/epoch
        means, logvar, var, norm = self.callEncZ(inp)

        output = self.decoder(norm)

        if len(self.outlayers) > 0:
            output = [layer(output) for layer in self.outlayers[0]]
            for level in self.outlayers[1:]: # TODO: Get rid of for loops
                output = [layer(i) for layer, i in zip(level, output)]
        outmean, outlogvar, outvar, outnorm = self.callEncZ(output)
        

        out = [tf.concat([means, logvar, var, outmean, outvar], axis = 1), *output, norm]

        return out

    def predict(self, inp, L = 50):
        """
        Reconstruction probability
        """
        probs = np.zeros([len(inp)])
        out = self.call(inp)
        y_pred = out[0]
        norm = out[3]

        mu_post = y_pred[:, self.latentsize*3:self.latentsize*4]
        var_post = y_pred[:, self.latentsize*4:self.latentsize*5]

        normal = tfp.distributions.Normal(mu_post, var_post)
        probs = normal.prob(norm)

        for l in range(1, L):
            post = self.callEncZ(out[1:3])
            mu_post = post[0]
            var_post = post[1]

            normal = tfp.distributions.Normal(mu_post, var_post)
            probs += normal.prob(norm)
        probs = np.nanmean(probs, axis = 1)
        return [np.expand_dims( 1 - (probs/L), -1)]


    def info(self):
        VAEargs = ['inputsize', 'inlayersize', 'latentsize', 'outlayersize', 'outputsize']
        res = {}
        for arg in VAEargs:
            res[arg] = self.__dict__[arg]
        return res

    def addcompile(self, fn):
        self.compile_fn = fn

    def _compile(self):
        self.compile_fn(self)


class VAEdistance(VAE):
    """
    Just VAE with special predict function
    """
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
        super(VAEdistance, self).__init__(inputsize, inlayersize, latentsize, outlayersize, outputsize, finalactivation)

    def predict(self, inp):
        return [np.mean(np.square(np.asarray(inp) - np.asarray(self.call(inp)[:-1])), 0)]

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
        output.append(output[-1]) #As the model only reconstructs the point, we add filler data and ignore it, because the class is still needed in the testing set to calculate metrics/confusion matrix.
        return output
    

class VAErcp3(VAEdistance):
    """
    Old naming used for loading saves
    """
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
        super(VAErcp3, self).__init__(inputsize, inlayersize, latentsize, outlayersize, outputsize, finalactivation)
