import tensorflow as tf
from keras import backend as K
from itertools import product
from functools import partial

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

def testloss(labels, predictions):
    e1 = tf.keras.losses.mean_squared_error(labels[0], predictions[0])
    e2 = tf.keras.losses.mean_squared_error(labels[1], predictions[1])
    return tf.math.reduce_mean([e1,e2])

def test1():
    matplotlib.get_backend()
    x1 = np.random.multivariate_normal([10,24], [[3,2],[2,3]], 100).astype('float32')
    x2 = np.random.multivariate_normal([-3,30], [[5,3],[3,6]], 100).astype('float32')
    x3 = np.random.multivariate_normal([-30,3], [[4,3],[3,4]], 100).astype('float32')
    x4 = np.array(list(zip(np.random.uniform(-50,50,10), np.random.uniform(-50,50,10)))).astype('float32')
    X = np.concatenate([x1, x2, x3, x4])
    Y = np.concatenate([np.zeros(len(x1)), np.zeros(len(x2)), np.zeros(len(x3)), np.ones(len(x4))])
    plt.plot(X[Y==0,0], X[Y==0,1], '.b')
    plt.plot(X[Y==1,0], X[Y==1,1], 'xb')
    Y = np.expand_dims(Y,-1).astype('float32')
    #print(list(zip(np.expand_dims(X,1),Y)))

    vae = VAE([],[128,64,16], 8, outlayersize=[16,64,128], outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    z = np.array(list(zip(X,Y)), dtype=object)
    traindata, test = splitdata(X,z, seed = 1)
    trainin, trainout = traindata[:,0], traindata[:,1]
    testin, testout = test[:,0], test[:,1]
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    trainout = [np.stack(trainout[:,0]).astype('float32'),np.stack(trainout[:,1]).astype('float32')]
    testout = [np.stack(testout[:,0]).astype('float32'),np.stack(testout[:,1]).astype('float32')]

    vae.compile(optimizer, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    hist = vae.fit(trainin.astype('float32'), trainout, validation_data = (testin.astype('float32'), testout),epochs=2000, batch_size=64, verbose = 2)
    #train(vae, trainin.astype('float32'), [np.stack(trainout[:,0]).astype('float32'),np.stack(trainout[:,1]).astype('float32')], lossfunction = testloss)



    x1 = np.random.multivariate_normal([10,24], [[3,2],[2,3]], 100).astype('float32')
    x2 = np.random.multivariate_normal([-3,30], [[5,3],[3,6]], 100).astype('float32')
    x3 = np.random.multivariate_normal([-30,3], [[4,3],[3,4]], 100).astype('float32')
    x4 = np.array(list(zip(np.random.uniform(-50,50,100), np.random.uniform(-50,50,100)))).astype('float32')
    X = np.concatenate([x1, x2, x3, x4])
    Y = np.concatenate([np.zeros(len(x1)), np.zeros(len(x2)), np.zeros(len(x3)), np.ones(len(x4))])

    pred = vae.predict(X)
    pred = np.where(np.array(pred[1]) >= 0.5, 1, 0)
    cm = np.zeros([2,2])
    pred = np.squeeze(pred,1)
    plt.plot(X[np.logical_and(Y==0, Y != pred),0], X[np.logical_and(Y==0, Y != pred),1], '.r')
    plt.plot(X[np.logical_and(Y==1, Y != pred),0], X[np.logical_and(Y==1, Y != pred),1], 'xr')
    plt.plot(X[np.logical_and(Y==0, Y == pred),0], X[np.logical_and(Y==0, Y == pred),1], '.k')
    plt.plot(X[np.logical_and(Y==1, Y == pred),0], X[np.logical_and(Y==1, Y == pred),1], 'xk')

    np.add.at(cm, (pred, Y.astype(int)), 1)
    
    vae.summary()
    plt.show()

def testdata(n1=100,n2=100,n3=100,n4=10):
    x1 = np.random.multivariate_normal([10,24], [[3,2],[2,3]], n1).astype('float32')
    x2 = np.random.multivariate_normal([-3,30], [[5,3],[3,6]], n2).astype('float32')
    x3 = np.random.multivariate_normal([-30,3], [[4,3],[3,4]], n3).astype('float32')
    x4 = np.array(list(zip(np.random.uniform(-50,50,n4), np.random.uniform(-50,50,n4)))).astype('float32')
    X = np.concatenate([x1, x2, x3, x4])
    Y = np.concatenate([np.zeros(len(x1)), np.zeros(len(x2)), np.zeros(len(x3)), np.ones(len(x4))])
    Y = np.expand_dims(Y,-1).astype('float32')
    return X,Y

def mptraining(a1, a2):
    vae = VAE(**a1)
    vae.compile(tf.keras.optimizers.Adam(learning_rate=1e-3), loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    a2['model'] = vae
    modeltests(**a2)

def testp1(X,Y,Xtest, Ytest):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    vae = VAE([],[128,64,16], 8, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '01', description='128|64|16, Latent of 8, 500 epochs against 100 from each distribution and 10 outliers')

    vae = VAE([],[64,16], 8, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '02', description='64|16, Latent of 8, 500 epochs against 100 from each distribution and 10 outliers')

    vae = VAE([],[64,16], 4, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '03', description='64|16, Latent of 4, 500 epochs against 100 from each distribution and 10 outliers')

    vae = VAE([],[64,32], 16, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '04', description='64|16, Latent of 16 500 epochs against 100 from each distribution and 10 outliers')

    vae = VAE([],[256,128,64], 16, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '05', description='256|128|64, Latent of 16, 500 epochs against 100 from each distribution and 10 outliers')

def testp2(X,Y,Xtest,Ytest):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    vae = VAE([],[128,64,16], 8, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '11', description='128|64|16, Latent of 8, 500 epochs against 200 from each distribution and 20 outliers')

    vae = VAE([],[64,16], 8, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '12', description='64|16, Latent of 8, 500 epochs against 200 from each distribution and 20 outliers')

    vae = VAE([],[64,16], 4, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '13', description='64|16, Latent of 4, 500 epochs against 200 from each distribution and 20 outliers')

    vae = VAE([],[64,32], 16, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '14', description='64|32, Latent of 16 500 epochs against 200 from each distribution and 20 outliers')

    vae = VAE([],[256,128,64], 16, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '15', description='256|128|64, Latent of 16, 500 epochs against 200 from each distribution and 20 outliers')

def loss3(true, predict, ce = None):
    print('true')
    print(predict)
    loss1 = tf.keras.losses.mean_squared_error(true[:,0], predict[:,0])
    loss2 = ce(true[:,1], predict[:,1])
    return loss1, loss2

def testp3(X,Y,Xtest,Ytest):
    from train import weightedce
    ce = partial(weightedce, weights = tf.constant([[1,10],[1,1]], dtype = tf.float32))
    loss = [tf.keras.losses.mean_squared_error, ce]
    ce.__name__ = 'weightedce'
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    vae = VAE([],[128,64,16], 8, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=loss)
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '31', verbose= 2, 
        description='128|64|16, Latent of 8, 500 epochs against 200 from each distribution and 20 outliers with weighted loss with tf.function tag')

    vae = VAE([],[64,16], 8, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=loss)
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '32', 
        description='64|16, Latent of 8, 500 epochs against 200 from each distribution and 20 outliers with weighted loss with tf.function tag')

    vae = VAE([],[64,16], 4, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=loss)
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '33', 
        description='64|16, Latent of 4, 500 epochs against 200 from each distribution and 20 outliers with weighted loss with tf.function tag')

    vae = VAE([],[64,32], 16, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=loss)
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '34', 
        description='64|32, Latent of 16 500 epochs against 200 from each distribution and 20 outliers with weighted loss with tf.function tag')

    vae = VAE([],[256,128,64], 16, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=loss)
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '35', 
        description='256|128|64, Latent of 16, 500 epochs against 200 from each distribution and 20 outliers with weighted loss with tf.function tag')

def testp4(X,Y,Xtest,Ytest):
    from train import weightedce
    ce = partial(weightedce, weights = tf.constant([[1,10],[1,1]], dtype = tf.float32))
    loss = [tf.keras.losses.mean_squared_error, ce]
    ce.__name__ = 'weightedce'
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    vae = VAE([],[128,64,16], 8, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=loss)
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '41', verbose= 2, 
        description='128|64|16, Latent of 8, 500 epochs against 200 from each distribution and 20 outliers with weighted loss with tf.function tag')

    vae = VAE([],[64,16], 8, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=loss)
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '42', 
        description='64|16, Latent of 8, 500 epochs against 200 from each distribution and 20 outliers with weighted loss with tf.function tag')

    vae = VAE([],[64,16], 4, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=loss)
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '43', 
        description='64|16, Latent of 4, 500 epochs against 200 from each distribution and 20 outliers with weighted loss with tf.function tag')

    vae = VAE([],[64,32], 16, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=loss)
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '44', 
        description='64|32, Latent of 16 500 epochs against 200 from each distribution and 20 outliers with weighted loss with tf.function tag')

    vae = VAE([],[256,128,64], 16, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=loss)
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '45', 
        description='256|128|64, Latent of 16, 500 epochs against 200 from each distribution and 20 outliers with weighted loss with tf.function tag')

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from train import train, splitdata, crossvalidation, modeltests
    import matplotlib
    import random
    import multiprocessing

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