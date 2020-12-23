import tensorflow as tf

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
            outlayersize = reversed(inlayersize)

        if outputsize == None:
            outputsize = reversed(inputsize)

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

        inp = tf.concat(inp, 0)
        means, logvar = self.encoder(inp)
        
        var = tf.math.exp(logvar)
        
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

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from train import train, splitdata

    x1 = np.random.multivariate_normal([10,24], [[3,2],[2,3]], 100).astype('float32')
    x2 = np.random.multivariate_normal([-3,30], [[5,3],[3,6]], 100).astype('float32')
    x3 = np.random.multivariate_normal([-30,3], [[4,3],[3,4]], 100).astype('float32')
    x4 = np.array(list(zip(np.random.uniform(-50,50,10), np.random.uniform(-50,50,10)))).astype('float32')
    X = np.concatenate([x1, x2, x3, x4])
    Y = np.concatenate([np.zeros(len(x1)), np.zeros(len(x2)), np.zeros(len(x3)), np.ones(len(x4))])
    Y = np.expand_dims(Y,-1).astype('float32')
    #print(list(zip(np.expand_dims(X,1),Y)))

    vae = VAE([],[32,16,8], 4, outlayersize= [8,16,32],outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    z = np.array(list(zip(X,Y)), dtype=object)
    traindata, test = splitdata(X,z)
    trainin, trainout = traindata[:,0], traindata[:,1]
    testin, testout = test[:,0], test[:,1]
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    vae.compile(optimizer, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    vae.fit(trainin.astype('float32'), [np.stack(trainout[:,0]).astype('float32'),np.stack(trainout[:,1]).astype('float32')], epochs=500, batch_size=64, verbose = 2)
    #train(vae, trainin.astype('float32'), [np.stack(trainout[:,0]).astype('float32'),np.stack(trainout[:,1]).astype('float32')], lossfunction = testloss)
    pred = vae.predict(testin.astype('float32'))
    pred = np.where(np.array(pred[1]) >= 0.5, 0, 1)
    print(list(zip(np.stack(testout[:,1]),np.stack(pred))))
    cm = np.zeros([2,2])
    np.add.at(cm, [np.stack(testout[:,1]).astype('int'),pred], 1)
    print(cm)