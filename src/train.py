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

def splitdata(inp, labels, testsplit = 0.2, seed = 0):
    np.random.seed(seed)
    inp, labels = np.array(list(zip(*inp))), np.array(list(zip(*labels)))
    z = np.array(list(zip(inp, labels)))
    np.random.shuffle(z)

    ind = int(len(inp)*(1 - testsplit))
    train, test = z[:ind], z[ind:]
    trainin, trainout = rowtocols(train[:,0]), rowtocols(train[:,1])
    testin, testout = rowtocols(test[:,0]), rowtocols(test[:,1])

    return (trainin, trainout), (testin, testout)

def rowtocols(inp):
    res = []
    for i in range(len(inp[0])):
        res.append(np.stack(np.vstack(inp)[:,i]))
    return res

def cvcol(inp):
    res = []
    for t in inp:
        res.append(rowtocols(t))
    return np.asarray(res)

def joinfolds(folds):
    stacked = np.column_stack(folds)
    return [np.vstack(stacked[x]) for x in range(len(stacked))]

def kfoldstratify(labels, k):
    labels = labels.reshape([len(labels)])
    class1 = np.squeeze(np.where(labels==1))
    class0 = np.squeeze(np.where(labels==0))
    w = [np.concatenate(x) for x in zip(np.array_split(class0,k), 
                                reversed(np.array_split(class1,k)))]
    [np.random.shuffle(x) for x in w]

    return np.array(w)

def crossvalidation(inp, labels, model, epochs=500, k = 5, seed = 0, verbose = 0):
    results = {'acc':[], 'loss':[], 'cm':[], 'spec':[]}
    
    fold_ind = kfoldstratify(labels[-1], k)

    for i in range(k):
        model.reset_metrics()
        arr = np.setdiff1d(list(range(k)), [i]).astype(int)
        testind, trainind = np.concatenate(fold_ind[arr]), fold_ind[i]

        ktrainout = [col[trainind] for col in labels]
        ktrainin = [col[trainind] for col in inp]
        
        ktestout = [col[testind] for col in labels]
        ktestin = [col[testind] for col in inp]
        
        hist = model.fit(ktrainin, ktrainout, epochs = epochs, verbose=verbose).history

        pred = model.predict(ktestin)
        pred = np.where(np.array(pred[-1]) >= 0.5, 1, 0)
        Y = ktestout[-1]
        cm = np.zeros([2,2])
        np.add.at(cm, (pred, Y.astype(int)), 1)
        results['loss'].append([hist['loss'][-1],hist['output_12_loss'][-1]])
        results['acc'].append((cm[0][0] + cm[1][1])/cm.sum())
        results['spec'].append(cm[1][1]/(cm[:,1].sum()))
        results['cm'].append(cm)

    return results

def modeltests(inp, labels, testdata, model, name, description = None, epochs=500, k = 5, seed = 0, verbose = 0, testsplit = 0.2, dir = 'src/reports/test1', plot = False):
    import matplotlib.pyplot as plt
    import time
    np.random.seed(seed)

    start = time.perf_counter()
    cv = crossvalidation(inp, labels, model, epochs, k, seed, verbose = verbose)
    elapsed = time.perf_counter() - start
    a = np.mean(cv['cm'],axis = 0)
    ave = (a[0][0] + a[1][1])/a.sum()
    spec = a[1][1]/(a[:,1].sum())
    loss = np.mean(cv['loss'],0)
    astr = f'||True 0| True 1|\n|-|-|-|\n|Predicted 0|{a[0][0]}|{a[0][1]}\n|Predicted 1|{a[1][0]}|{a[1][1]}\n'
    print(name)
    print(astr)
    print(f'Average accuracy: {ave}')
    print(f'Average specificity: {spec}')
    print(f'Average loss: {loss}')
    print(f'|Acc|Spec|Loss|\n{ave}|{spec}|{loss}')
    print(f'Took {elapsed} seconds')

    name = f'{dir}/{name}'

    model.reset_metrics()

    if testdata == None:
        (inp, labels), testdata = splitdata(inp, labels, testsplit, seed)
    start = time.perf_counter()
    model.fit(inp, labels, epochs=epochs, batch_size=64, verbose = verbose)
    elapsedtest = time.perf_counter() - start

    if plot:
        X,Y = labels

        Y = np.hstack(Y)

        plt.plot(X[Y==0,0], X[Y==0,1], '.b')
        plt.plot(X[Y==1,0], X[Y==1,1], 'xb')

    X,Y = testdata

    pred = model.predict(X)
    pred = np.where(np.array(pred[-1]) >= 0.5, 1, 0)
    Y = Y[-1]
    tcm = np.zeros([2,2])
    pred = np.squeeze(pred,1)
    np.add.at(tcm, (pred, Y.astype(int)), 1)

    with open(f'{name}', 'w') as file:
        #file.write(name)
        if description is not None:
            file.write(f'{description}\n')
        file.writelines('\n'.join([astr, f'Average accuracy: {ave}', f'Average specificity: {spec}', 
        f'Average loss: {loss}', f'Average cm: {a}', f'|Acc|Spec|Loss|\n{ave}|{spec}|{loss}',f'Test cm: {tcm}', 
        f'CV took {elapsed} seconds', f'Fitting all data took {elapsedtest} seconds', 
        f'||True 0| True 1|\n|-|-|-|\n|Predicted 0|{tcm[0][0]}|{tcm[0][1]}\n|Predicted 1|{tcm[1][0]}|{tcm[1][1]}\n']))

    if plot:
        plt.plot(X[np.logical_and(Y==0, Y != pred),0], X[np.logical_and(Y==0, Y != pred),1], '.r')
        plt.plot(X[np.logical_and(Y==1, Y != pred),0], X[np.logical_and(Y==1, Y != pred),1], 'xr')
        plt.plot(X[np.logical_and(Y==0, Y == pred),0], X[np.logical_and(Y==0, Y == pred),1], '.k')
        plt.plot(X[np.logical_and(Y==1, Y == pred),0], X[np.logical_and(Y==1, Y == pred),1], 'xk')
        plt.savefig(f'{name}.png')

@tf.function
def weightedce(y_true, y_pred, weights, conf = 0.5):    
    conf = tf.constant(conf)

    predclass = tf.where(tf.less(tf.squeeze(y_pred), conf), tf.constant(0,dtype=tf.int32), tf.constant(1,dtype=tf.int32))
    predclass = tf.gather(weights, predclass)

    mask = tf.where(tf.math.equal(tf.squeeze(y_true), tf.constant(0.)), predclass[:,0], predclass[:,1])

    return tf.math.multiply(tf.keras.losses.binary_crossentropy(y_true, y_pred), mask)