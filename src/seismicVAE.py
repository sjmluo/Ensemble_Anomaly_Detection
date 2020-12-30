import tensorflow as tf
from VAE import VAE
from train import train_step

def lossfunction(labels, predictions):
    # Calculate loss across all labels then provide weighted loss towards the class predictor

    gdenergy = tf.keras.losses.MSE(labels[0], predictions[0])
    gdpuls = tf.keras.losses.MSE(labels[1], predictions[1])
    genergy = tf.keras.losses.MSE(labels[2], predictions[2])
    energy = tf.keras.losses.MSE(labels[3], predictions[3])
    maxenergy = tf.keras.losses.MSE(labels[4], predictions[4])
    gpuls = tf.keras.losses.MSE(labels[5], predictions[5])
    shift = tf.keras.losses.categorical_crossentropy(labels[6], predictions[6])
    seismic = tf.keras.losses.categorical_crossentropy(labels[7], predictions[7])
    seismoacoustic = tf.keras.losses.categorical_crossentropy(labels[8], predictions[8])
    ghazard = tf.keras.losses.categorical_crossentropy(labels[9], predictions[9])
    nbumpsv = tf.keras.losses.categorical_crossentropy(labels[10], predictions[10])
    classp = tf.keras.losses.binary_crossentropy(labels[11], predictions[11])
    weights = [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.115, 0.115, 0.115, 0.115, 0.3]
    indivloss = [gdenergy, gdpuls, genergy, energy, maxenergy, gpuls, shift, seismic, seismoacoustic, ghazard, nbumpsv, classp]

    return tf.nn.compute_average_loss(indivloss, sample_weight=weights)

if __name__ == "__main__":
    from data_exploration.seismicdata import data
    from train import splitdata, modeltests

    vae = VAE([[1,1,1,1,1,1,2,4,4,4,4],[4,4,4,4,4,4,8,16,16,16,16]], [256, 128, 64], 16, 
        outputsize=[[4,4,4,4,4,4,8,16,16,16,16,4], [1,1,1,1,1,1,2,4,4,4,4,1]],
        finalactivation=[None,None,None,None,None,None,None,None,None,None,None,'sigmoid'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    losses = [tf.losses.mean_squared_error]*11
    losses.append(tf.losses.binary_crossentropy)
    vae.compile(optimizer, loss=losses, loss_weights=[1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 5])
    
    inp, tar = data()

    '''traindata, test = splitdata(inp, tar, seed = 0)
    trainin, trainout = traindata[:,0], traindata[:,1]
    testin, testout = test[:,0], test[:,1]'''
    modeltests(inp, tar, None, model = vae, epochs = 1, name = '03', 
    description="""VAE([[1,1,1,1,1,1,2,4,4,4,4],[4,4,4,4,4,4,8,16,16,16,16]], [256, 128, 64], 16, 
        outputsize=[[4,4,4,4,4,4,8,16,16,16,16,4], [1,1,1,1,1,1,2,4,4,4,4,1]],
        finalactivation=[None,None,None,None,None,None,None,None,None,None,None,'sigmoid'])
         500 epochs with bce with weighted losses: [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 5]"""
        , dir = 'src/reports/seismic', verbose=2)
