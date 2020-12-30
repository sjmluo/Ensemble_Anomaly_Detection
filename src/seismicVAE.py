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

    vae = VAE([[4,4,4,4,4,4,4,8,8,8,8],[8,8,8,8,8,8,16,32,32,32,32]], [516, 256, 128], 64, 
        outputsize=[[4,4,4,4,4,4,8,16,16,16,16,4], [1,1,1,1,1,1,2,4,4,4,4,1]],
        finalactivation=[None,None,None,None,None,None,None,None,None,None,None,'sigmoid'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    losses = [tf.losses.mean_absolute_error]*11
    losses.append(tf.losses.binary_crossentropy)
    vae.compile(optimizer, loss=losses, loss_weights=[1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1])
    
    earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_output_12_loss', min_delta=10e-7, patience=50, verbose=2,
    mode='auto', baseline=None, restore_best_weights=True
    )
    plateau = tf.keras.callbacks.ReduceLROnPlateau()
    logbar = tf.keras.callbacks.ProgbarLogger(stateful_metrics=['output_1_loss',
        'output_2_loss',
        'output_3_loss',
        'output_4_loss',
        'output_5_loss',
        'output_6_loss',
        'output_7_loss',
        'output_8_loss',
        'output_9_loss',
        'output_10_loss',
        'output_11_loss',
        'val_output_1_loss',
        'val_output_2_loss',
        'val_output_3_loss',
        'val_output_4_loss',
        'val_output_5_loss',
        'val_output_6_loss',
        'val_output_7_loss',
        'val_output_8_loss',
        'val_output_9_loss',
        'val_output_10_loss',
        'val_output_11_loss','loss'])
    #logbar.epochs = 25

    inp, tar = data()

    modeltests(inp, tar, None, model = vae, epochs = 500, name = '04', 
    description="""VAE([[4,4,4,4,4,4,4,8,8,8,8],[8,8,8,8,8,8,16,32,32,32,32]], [516, 256, 128], 64, 
        outputsize=[[4,4,4,4,4,4,8,16,16,16,16,4], [1,1,1,1,1,1,2,4,4,4,4,1]],
        finalactivation=[None,None,None,None,None,None,None,None,None,None,None,'sigmoid'])
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        losses = [tf.losses.mean_absolute_error]*11
        losses.append(tf.losses.binary_crossentropy)
        vae.compile(optimizer, loss=losses, loss_weights=[1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 10]), using mae instead of mse"""
        , dir = 'src/reports/seismic', verbose = 2,overall_kwargs = {'callbacks': [plateau,logbar]})
