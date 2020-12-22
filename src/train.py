import tensorflow as tf
from VAE import VAE

@tf.function
def train_step(model:tf.keras.Model, inp, lossfunction = tf.keras.losses.mean_squared_error, optimizer = tf.keras.optimizers.Adam()):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss = lossfunction(inp, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train(model:tf.keras.Model, inp, lossfunction = None, optimizer = None, epochs = 200, earlyexit = None):
    if loss == None: loss = tf.keras.losses.mean_squared_error
    if optimizer == None: optimizer = tf.keras.optimizers.Adam()
    
    losses = []
    for e in range(epochs):
        loss = train_step(model, inp, lossfunction, optimizer)
        losses.append(loss)

    return losses

if __name__ == "__main__":
    pass