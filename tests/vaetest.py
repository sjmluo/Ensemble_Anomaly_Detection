from src.train import weightedce
import numpy as np
from src.VAE import VAE
from src.train import modeltests
import tensorflow as tf
from functools import partial
import matplotlib
import matplotlib.pyplot as plt


def testdata(n1=100,n2=100,n3=100,n4=10):
    x1 = np.random.multivariate_normal([10,24], [[3,2],[2,3]], n1).astype('float32')
    x2 = np.random.multivariate_normal([-3,30], [[5,3],[3,6]], n2).astype('float32')
    x3 = np.random.multivariate_normal([-30,3], [[4,3],[3,4]], n3).astype('float32')
    x4 = np.array(list(zip(np.random.uniform(-50,50,n4), np.random.uniform(-50,50,n4)))).astype('float32')
    X = np.concatenate([x1, x2, x3, x4])
    Y = np.concatenate([np.zeros(len(x1)), np.zeros(len(x2)), np.zeros(len(x3)), np.ones(len(x4))])
    Y = np.expand_dims(Y,-1).astype('float32')
    return X,Y


def test1():
    matplotlib.get_backend()
    x1 = np.random.multivariate_normal([10,24], [[3,2],[2,3]], 100).astype('float32')
    x2 = np.random.multivariate_normal([-3,30], [[5,3],[3,6]], 100).astype('float32')
    x3 = np.random.multivariate_normal([-30,3], [[4,3],[3,4]], 100).astype('float32')
    x4 = np.array(list(zip(np.random.uniform(-50,50,10), np.random.uniform(-50,50,10)))).astype('float32')
    X = np.concatenate([x1, x2, x3, x4])
    Y = np.concatenate([np.zeros(len(x1)), np.zeros(len(x2)), np.zeros(len(x3)), np.ones(len(x4))])
    plt.plot(X[Y==0,0], X[Y==0,1], '.b')
    plt.plot(X[Y==1,0], X[Y==1,1], 'xb')
    Y = np.expand_dims(Y,-1).astype('float32')
    #print(list(zip(np.expand_dims(X,1),Y)))

    vae = VAE([],[128,64,16], 8, outlayersize=[16,64,128], outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    z = np.array(list(zip(X,Y)), dtype=object)
    traindata, test = splitdata(X,z, seed = 1)
    trainin, trainout = traindata[:,0], traindata[:,1]
    testin, testout = test[:,0], test[:,1]
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    trainout = [np.stack(trainout[:,0]).astype('float32'),np.stack(trainout[:,1]).astype('float32')]
    testout = [np.stack(testout[:,0]).astype('float32'),np.stack(testout[:,1]).astype('float32')]

    vae.compile(optimizer, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    hist = vae.fit(trainin.astype('float32'), trainout, validation_data = (testin.astype('float32'), testout),epochs=2000, batch_size=64, verbose = 2)
    #train(vae, trainin.astype('float32'), [np.stack(trainout[:,0]).astype('float32'),np.stack(trainout[:,1]).astype('float32')], lossfunction = testloss)



    x1 = np.random.multivariate_normal([10,24], [[3,2],[2,3]], 100).astype('float32')
    x2 = np.random.multivariate_normal([-3,30], [[5,3],[3,6]], 100).astype('float32')
    x3 = np.random.multivariate_normal([-30,3], [[4,3],[3,4]], 100).astype('float32')
    x4 = np.array(list(zip(np.random.uniform(-50,50,100), np.random.uniform(-50,50,100)))).astype('float32')
    X = np.concatenate([x1, x2, x3, x4])
    Y = np.concatenate([np.zeros(len(x1)), np.zeros(len(x2)), np.zeros(len(x3)), np.ones(len(x4))])

    pred = vae.predict(X)
    pred = np.where(np.array(pred[1]) >= 0.5, 1, 0)
    cm = np.zeros([2,2])
    pred = np.squeeze(pred,1)
    plt.plot(X[np.logical_and(Y==0, Y != pred),0], X[np.logical_and(Y==0, Y != pred),1], '.r')
    plt.plot(X[np.logical_and(Y==1, Y != pred),0], X[np.logical_and(Y==1, Y != pred),1], 'xr')
    plt.plot(X[np.logical_and(Y==0, Y == pred),0], X[np.logical_and(Y==0, Y == pred),1], '.k')
    plt.plot(X[np.logical_and(Y==1, Y == pred),0], X[np.logical_and(Y==1, Y == pred),1], 'xk')

    np.add.at(cm, (pred, Y.astype(int)), 1)
    
    vae.summary()
    plt.show()

def mptraining(a1, a2):
    vae = VAE(**a1)
    vae.compile(tf.keras.optimizers.Adam(learning_rate=1e-3), loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    a2['model'] = vae
    modeltests(**a2)

def testp1(X,Y,Xtest, Ytest):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    vae = VAE([],[128,64,16], 8, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '01', description='128|64|16, Latent of 8, 500 epochs against 100 from each distribution and 10 outliers')

    vae = VAE([],[64,16], 8, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '02', description='64|16, Latent of 8, 500 epochs against 100 from each distribution and 10 outliers')

    vae = VAE([],[64,16], 4, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '03', description='64|16, Latent of 4, 500 epochs against 100 from each distribution and 10 outliers')

    vae = VAE([],[64,32], 16, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '04', description='64|16, Latent of 16 500 epochs against 100 from each distribution and 10 outliers')

    vae = VAE([],[256,128,64], 16, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '05', description='256|128|64, Latent of 16, 500 epochs against 100 from each distribution and 10 outliers')

def testp2(X,Y,Xtest,Ytest):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    vae = VAE([],[128,64,16], 8, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '11', description='128|64|16, Latent of 8, 500 epochs against 200 from each distribution and 20 outliers')

    vae = VAE([],[64,16], 8, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '12', description='64|16, Latent of 8, 500 epochs against 200 from each distribution and 20 outliers')

    vae = VAE([],[64,16], 4, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '13', description='64|16, Latent of 4, 500 epochs against 200 from each distribution and 20 outliers')

    vae = VAE([],[64,32], 16, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '14', description='64|32, Latent of 16 500 epochs against 200 from each distribution and 20 outliers')

    vae = VAE([],[256,128,64], 16, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=[tf.keras.losses.mean_squared_error, tf.keras.losses.binary_crossentropy])
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '15', description='256|128|64, Latent of 16, 500 epochs against 200 from each distribution and 20 outliers')

def loss3(true, predict, ce = None):
    print('true')
    print(predict)
    loss1 = tf.keras.losses.mean_squared_error(true[:,0], predict[:,0])
    loss2 = ce(true[:,1], predict[:,1])
    return loss1, loss2

def testp3(X,Y,Xtest,Ytest):
    ce = partial(weightedce, weights = tf.constant([[1,10],[1,1]], dtype = tf.float32))
    loss = [tf.keras.losses.mean_squared_error, ce]
    ce.__name__ = 'weightedce'
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    vae = VAE([],[128,64,16], 8, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=loss)
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '31', verbose= 2, 
        description='128|64|16, Latent of 8, 500 epochs against 200 from each distribution and 20 outliers with weighted loss with tf.function tag')

    vae = VAE([],[64,16], 8, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=loss)
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '32', 
        description='64|16, Latent of 8, 500 epochs against 200 from each distribution and 20 outliers with weighted loss with tf.function tag')

    vae = VAE([],[64,16], 4, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=loss)
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '33', 
        description='64|16, Latent of 4, 500 epochs against 200 from each distribution and 20 outliers with weighted loss with tf.function tag')

    vae = VAE([],[64,32], 16, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=loss)
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '34', 
        description='64|32, Latent of 16 500 epochs against 200 from each distribution and 20 outliers with weighted loss with tf.function tag')

    vae = VAE([],[256,128,64], 16, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=loss)
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '35', 
        description='256|128|64, Latent of 16, 500 epochs against 200 from each distribution and 20 outliers with weighted loss with tf.function tag')

def testp4(X,Y,Xtest,Ytest):
    ce = partial(weightedce, weights = tf.constant([[1,10],[1,1]], dtype = tf.float32))
    loss = [tf.keras.losses.mean_squared_error, ce]
    ce.__name__ = 'weightedce'
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    vae = VAE([],[128,64,16], 8, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=loss)
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '41', verbose= 2, 
        description='128|64|16, Latent of 8, 500 epochs against 200 from each distribution and 20 outliers with weighted loss with tf.function tag')

    vae = VAE([],[64,16], 8, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=loss)
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '42', 
        description='64|16, Latent of 8, 500 epochs against 200 from each distribution and 20 outliers with weighted loss with tf.function tag')

    vae = VAE([],[64,16], 4, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=loss)
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '43', 
        description='64|16, Latent of 4, 500 epochs against 200 from each distribution and 20 outliers with weighted loss with tf.function tag')

    vae = VAE([],[64,32], 16, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=loss)
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '44', 
        description='64|32, Latent of 16 500 epochs against 200 from each distribution and 20 outliers with weighted loss with tf.function tag')

    vae = VAE([],[256,128,64], 16, outputsize = [[2,1]], finalactivation=[None, 'sigmoid'])
    vae.compile(optimizer, loss=loss)
    modeltests(X,[X,Y], [Xtest, Ytest], model = vae, epochs = 500, name = '45', 
        description='256|128|64, Latent of 16, 500 epochs against 200 from each distribution and 20 outliers with weighted loss with tf.function tag')