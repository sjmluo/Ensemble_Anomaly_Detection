import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self, layersizes, latentsize):
        super().__init__()

        self.mlp = [tf.keras.layers.Dense(layer,activation = 'relu') for layer in layersize]
       
        self.means = tf.keras.layers.Dense(latentsize)
        self.logvar = tf.keras.layers.Dense(latentsize)

    def call(self, inp):
        x = inp
        for layer in self.mlp:
            x = layer(x)
        
        return self.means(x), self.logvar(x)
