import tensorflow as tf
from VAE import VAE
import numpy as np

#@tf.function
def train_step(model:tf.keras.Model, inp, labels, lossfunction, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss = lossfunction(labels, predictions)
        print(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train(model:tf.keras.Model, inp, labels, lossfunction = None, optimizer = None, epochs = 200, earlyexit = None):
    if lossfunction == None: lossfunction = tf.keras.losses.mean_squared_error
    if optimizer == None: optimizer = tf.keras.optimizers.Adam()

    losses = []
    for e in range(epochs):
        print(f'Training epoch {e}', end = '\r')
        loss = train_step(model, inp, labels, lossfunction, optimizer)
        losses.append(loss)
        print(f'Epoch {e} had loss of {loss}')

    return losses

def splitdata(inp, output, testsplit = 0.2, seed = 0):
    np.random.seed(0)
    z = np.asarray(list(zip(inp, output)))
    np.random.shuffle(z)
    ind = int(len(inp)*(1 - testsplit))
    return z[:ind], z[ind:]