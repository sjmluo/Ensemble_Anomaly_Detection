import tensorflow as tf
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
            print(layer)
            x = layer(x)
        
        return self.means(x), self.logvar(x)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, layersizes, outputsize):
        super(Decoder, self).__init__()

        self.mlp = [tf.keras.layers.Dense(layer,activation = 'relu') for layer in layersizes]
        print(outputsize)
        self.end = tf.keras.layers.Dense(outputsize)

    def call(self, inp):
        x = inp
        for layer in self.mlp:
            print(layer)
            x = layer(x)
        
        return self.end(x)

class VAE(tf.keras.Model):
    def __init__(self, inlayersize, outlayersize, latentsize, outputsize):
        super().__init__()
        self.encoder = Encoder(inlayersize, latentsize)
        self.decoder = Decoder(outlayersize, outputsize)

    def call(self, inp):
        means, logvar = self.encoder(inp)
        
        var = tf.math.exp(logvar)
        
        norm = tf.random.normal([1], means, var)

        output = self.decoder(norm)
        return output

if __name__ == "__main__":
    vae = VAE([2],[2],1,2)
    print(vae(np.array([[1,1],[2,2]])))
