import imblearn
import tensorflow as tf
import os
import numpy as np
import pandas as pd

def colstorows(inp):
    """
    Takes a list of different columns(inputs to model) and converts them to row-wise inputs
    #TODO a transpose might work faster?
    """
    df = pd.DataFrame()
    for i,c in enumerate(inp):
        df[i] = list(c)
    return df.to_numpy()

def splitdata(inp, labels, testsplit = (2,10), seed = 0):
    """
    Splits data into 2 pieces, where test is testsplit[0] out of testsplit[1] pieces.
    """
    if testsplit[0] == 1: #quick bodge
        testsplit[0] *=2
        testsplit[1] *=2
    np.random.seed(seed)
    fold_ind = kfoldstratify(labels[-1], testsplit[1])
    arr = np.setdiff1d(list(range(testsplit[1])), list(range(testsplit[0]))).astype(int)
    trainind, testind = np.concatenate(fold_ind[arr]), np.concatenate(fold_ind[list(range(testsplit[0]))])
    ktrainout = [col[trainind] for col in labels]
    ktrainin = [col[trainind] for col in inp]
    
    ktestout = [col[testind] for col in labels]
    ktestin = [col[testind] for col in inp]


    return (ktrainin, ktrainout), (ktestin, ktestout)

def rowtocols(inp):
    """
    Converts row-wise np array to a column-wise one
    """
    res = []
    for i in range(len(inp[0])):
        res.append(np.stack(np.vstack(inp)[:,i]))
    return res

def joinfolds(folds):
    """
    Joins multiple folds into one array
    """
    stacked = np.column_stack(folds)
    return [np.vstack(stacked[x]) for x in range(len(stacked))]

def kfoldstratify(labels, k, seed = None):
    """
    Takes in labels ONLY and returns k partitions of **indices**, ensuring binary classes are as evenly distributed as possible
    """
    if isinstance(seed,int): np.random.seed(seed)
    labels = np.reshape(np.asarray(labels).astype('int32'), [len(labels)])
    class1 = np.squeeze(np.where(labels==1))
    class0 = np.squeeze(np.where(labels==0))
    np.random.shuffle(class1)
    np.random.shuffle(class0)
    w = [np.concatenate(x) for x in zip(np.array_split(class0,k), 
                                reversed(np.array_split(class1,k)))]
    [np.random.shuffle(x) for x in w]

    return np.array(w, dtype='object')

def reset_model(model):
    """
    Manually resets each layer
    """
    for ix, layer in enumerate(model.layers):
        if hasattr(model.layers[ix], 'kernel_initializer') and \
                hasattr(model.layers[ix], 'bias_initializer'):
            weight_initializer = model.layers[ix].kernel_initializer
            bias_initializer = model.layers[ix].bias_initializer
            if len(layer.get_weights()) != 2:
                print('oops')
                continue
            old_weights, old_biases = model.layers[ix].get_weights()

            model.layers[ix].set_weights([
                weight_initializer(shape=old_weights.shape),
                bias_initializer(shape=len(old_biases))])

def confusionmat(pred, true, conf = 0.5):
    """
    Takes in prediction and true labels and produces confusion matrix, where a prediction of < conf is class 0, >= is class 1
    """
    pred = np.squeeze(pred,1)
    pred = np.where(np.array(pred) >= conf, 1, 0)
    true = np.squeeze(true).astype(int)
    cm = np.zeros([2,2])
    np.add.at(cm, (pred, true.astype(int)), 1)
    return cm

@tf.function
def weightedce(y_true, y_pred, weights = [[1,1],[1,1]], conf = 0.5):
    """
    Weighted binary crossentropy, weights is a 2D array corresponding to each type of prediction
    |      |Class 0|Class 1|
    |------|-------|-------|
    |Pred 0| [0][0]| [0][1]|
    |Pred 1| [1][0]| [1][1]|
    Each entry corresponds to index of weights argument
    Prediction is categories as class 0 for a prediction of < conf
    """
    conf = tf.constant(conf)

    predclass = tf.where(tf.less(tf.squeeze(y_pred), conf), tf.constant(0,dtype=tf.int32), tf.constant(1,dtype=tf.int32))
    predclass = tf.gather(weights, predclass)

    mask = tf.where(tf.math.equal(tf.squeeze(y_true), tf.constant(0)), predclass[:,0], predclass[:,1])

    return tf.math.multiply(tf.keras.losses.binary_crossentropy(y_true, y_pred), mask)

def bestalpha(y_true, y_pred):
    highestalpha = 0
    highestf1 = 0
    for alpha in sorted(np.squeeze(y_pred,-1)):
        cm = confusionmat(y_pred, y_true, alpha)
        f1 = 2*cm[1,1]/(2*cm[1,1]+cm[0,1]+cm[1,0])
        if f1 > highestf1:
            highestalpha = alpha
            highestf1 = f1
    return highestalpha, highestf1