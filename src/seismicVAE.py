import tensorflow as tf
from src.VAE import VAE
from src.train import train_step
from datetime import datetime

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


def callback(wdir):
    plateau = tf.keras.callbacks.ReduceLROnPlateau(verbose=1)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=wdir)

    earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_output_12_loss', min_delta=10e-7, patience=18, verbose=2,
    mode='auto', baseline=None, restore_best_weights=True
    )

    return [plateau, tb_callback, earlystop]
    
def compile_model(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    losses = [tf.losses.mean_absolute_error]*11
    #losses = []
    from src.train import weightedce
    from functools import partial
    ce = partial(weightedce, weights = tf.constant([[1,1],[1,1]], dtype = tf.float32))
    #losses.append(tf.losses.binary_crossentropy)
    losses.append(ce)
    ce.__name__ = 'weightedce'
    model.compile(optimizer, loss=losses, loss_weights=[1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 10])

if __name__ == "__main__":
    from src.data_exploration.seismicdata import data, rawdf, data_preprocessing
    from src.train import splitdata, modeltests, rowtocols
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.combine import SMOTETomek
    from sklearn.datasets import make_classification
    import numpy as np

    np.random.seed(0)

    vae = VAE([[4,4,4,4,4,4,4,8,8,8,8],[8,8,8,8,8,8,16,32,32,32,32]], [512, 512, 1024], 128, 
        outputsize=[[4,4,4,4,4,4,8,16,16,16,16,4], [1,1,1,1,1,1,2,4,4,4,4,1]],
        finalactivation=[None,None,None,None,None,None,None,None,None,None,None,'sigmoid'])
    
    vae.addcompile(compile_model)
    vae._compile()

    raw = rawdf()

    preprocessing = SMOTETomek(n_jobs=-1)

    train, test = splitdata([np.array(list(range(raw.shape[0])))], [raw.iloc[:,-1].to_numpy().astype('int32')], testsplit = (2,40))
    train = train[0][0]
    test = test[0][0]
    train, tar = preprocessing.fit_resample(raw.iloc[train,:-1], raw.iloc[train,-1].astype('int32'))
    train['class'] = tar
    train = data_preprocessing(train)
    test = raw.iloc[test,:].copy()
    test = data_preprocessing(test)


    X,y = preprocessing.fit_resample(raw.iloc[:,:-1], raw.iloc[:,-1].astype('int32'))
    X['class'] = y
    inp, tar = data_preprocessing(X)
    
    
    """ inp, tar = oversample.fit_resample(inp, np.squeeze(tar[:,-1]).astype('int32'))
    inp = rowtocols(inp)
    tar = list(inp) + [tar] """

    #vae.fit(train[0], train[1], validation_data = test)

    modeltests(inp, tar, (train, test), model = vae, epochs = 500, name = '23', k=10, 
    description="""VVAE([[4,4,4,4,4,4,4,8,8,8,8],[8,8,8,8,8,8,16,32,32,32,32]], [512, 512, 1024], 128, 
        outputsize=[[4,4,4,4,4,4,8,16,16,16,16,4], [1,1,1,1,1,1,2,4,4,4,4,1]],
        finalactivation=[None,None,None,None,None,None,None,None,None,None,None,'sigmoid'])
         with weightedce[[1,1],[1,1]],
         loss_weights=[1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 10], 10fold cv with naive SMOTETomek at the start
         , while splitting final testing data from preprocessing"""
        , wdir = 'src/reports/seismic', verbose = 2,overall_kwargs = {'callbacks': callback})
