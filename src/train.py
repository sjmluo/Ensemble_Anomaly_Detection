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

def crossvalidation(inp, labels, model, epochs=500, k = 5, seed = 0, verbose = 0):
    results = {'acc':[], 'loss':[], 'cm':[], 'spec':[]}
    
    np.random.seed(seed)
    #labels = [np.expand_dims(l,1) for l in labels]
    z = np.array(list(zip(*labels)))
    z = np.array(list(zip(inp, z)))
    np.random.shuffle(z)
    inp = np.array(np.array_split(np.stack(z[:,0]),5))
    #print(z[:,1])
    labels = np.array(np.array_split(np.stack(z[:,1]),5))

    for i in range(k):
        model.reset_metrics()
        arr = np.setdiff1d(list(range(k)), [i]).astype(int)
        ktrainout = np.vstack(np.vstack(labels[arr]))
        
        ktrainin, ktrainout = np.asarray(np.vstack(inp[arr])).astype('float32'), [np.vstack(ktrainout[:,0]), np.vstack(ktrainout[:,1])]
        
        ktestout = np.vstack(labels[i])
        ktestin, ktestout = np.vstack(inp[i]).astype('float32'), [np.vstack(ktestout[:,0]), np.vstack(ktestout[:,1])]

        hist = model.fit(ktrainin, ktrainout, epochs = epochs, verbose=0).history

        pred = model.predict(ktestin)
        pred = np.where(np.array(pred[1]) >= 0.5, 1, 0)
        Y = ktestout[1]
        cm = np.zeros([2,2])
        np.add.at(cm, (pred, Y.astype(int)), 1)
        results['loss'].append([hist['loss'][-1],hist['output_1_loss'][-1],hist['output_2_loss'][-1]])
        results['acc'].append((cm[0][0] + cm[1][1])/cm.sum())
        results['spec'].append(cm[1][1]/(cm[:,1].sum()))
        results['cm'].append(cm)

    return results

def modeltests(inp, labels, testdata, model, name, description = None, epochs=500, k = 5, seed = 0, verbose = 0):
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

    name = f'src/reports/test1/{name}'

    model.reset_metrics()

    start = time.perf_counter()
    model.fit(inp, labels, epochs=epochs, batch_size=64, verbose = verbose)
    elapsedtest = time.perf_counter() - start

    X,Y = labels

    Y = np.hstack(Y)

    plt.plot(X[Y==0,0], X[Y==0,1], '.b')
    plt.plot(X[Y==1,0], X[Y==1,1], 'xb')

    X,Y = testdata

    pred = model.predict(X)
    pred = np.where(np.array(pred[1]) >= 0.5, 1, 0)
    Y = np.hstack(Y)
    tcm = np.zeros([2,2])
    pred = np.squeeze(pred,1)
    np.add.at(tcm, (pred, Y.astype(int)), 1)

    with open(f'{name}', 'w') as file:
        #file.write(name)
        if description is not None:
            file.write(description)
        file.writelines('\n'.join([astr, f'Average accuracy: {ave}', f'Average specificity: {spec}', 
        f'Average loss: {loss}', f'Average cm: {a}', f'|Acc|Spec|Loss|\n{ave}|{spec}|{loss}',f'Test cm: {tcm}', 
        f'CV took {elapsed} seconds', f'Fitting all data took {elapsedtest} seconds', 
        f'||True 0| True 1|\n|-|-|-|\n|Predicted 0|{tcm[0][0]}|{tcm[0][1]}\n|Predicted 1|{tcm[1][0]}|{tcm[1][1]}\n']))

    plt.plot(X[np.logical_and(Y==0, Y != pred),0], X[np.logical_and(Y==0, Y != pred),1], '.r')
    plt.plot(X[np.logical_and(Y==1, Y != pred),0], X[np.logical_and(Y==1, Y != pred),1], 'xr')
    plt.plot(X[np.logical_and(Y==0, Y == pred),0], X[np.logical_and(Y==0, Y == pred),1], '.k')
    plt.plot(X[np.logical_and(Y==1, Y == pred),0], X[np.logical_and(Y==1, Y == pred),1], 'xk')
    plt.savefig(f'{name}.png')