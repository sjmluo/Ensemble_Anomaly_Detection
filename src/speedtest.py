import tensorflow as tf
import time
import numpy as np

def weightedceeager(y_true, y_pred, weights, conf = 0.5):
    weights = tf.constant(weights, dtype = tf.float32)
    def zero():return weights[0]
    def one(): return weights[1]
    
    conf = tf.constant(conf)
    
    def f(x):
        return tf.cond(x, zero, one)
    predclass = tf.map_fn(fn = f, elems = tf.less(tf.squeeze(y_pred), conf), fn_output_signature=tf.float32)
    mask = tf.where(tf.math.equal(y_true, tf.constant(0.)), predclass[:,0], predclass[:,1])

    return tf.keras.losses.binary_crossentropy(y_pred, y_true) * mask


def weightedce10(y_true, y_pred, weights, conf = 0.5):
    weights = tf.constant(weights, dtype = tf.float32)
    def zero():return weights[0]
    def one(): return weights[1]
    
    conf = tf.constant(conf)
    @tf.function
    def f(x):
        return tf.cond(x, zero, one)
    predclass = tf.map_fn(fn = f, elems = tf.less(tf.squeeze(y_pred), conf), fn_output_signature=tf.float32)
    mask = tf.where(tf.math.equal(y_true, tf.constant(0.)), predclass[:,0], predclass[:,1])

    return tf.keras.losses.binary_crossentropy(y_pred, y_true) * mask

def weightedce1(y_true, y_pred, weights, conf = 0.5):
    weights = tf.constant(weights, dtype = tf.float32)
    def zero():return weights[0]
    def one(): return weights[1]
    
    conf = tf.constant(conf)
    @tf.function
    def f(x):
        return tf.cond(x, zero, one)
    predclass = tf.map_fn(fn = f, elems = tf.less(tf.squeeze(y_pred), conf), fn_output_signature=tf.float32, parallel_iterations=1)
    mask = tf.where(tf.math.equal(y_true, tf.constant(0.)), predclass[:,0], predclass[:,1])

    return tf.keras.losses.binary_crossentropy(y_pred, y_true) * mask

def weightedce5(y_true, y_pred, weights, conf = 0.5):
    weights = tf.constant(weights, dtype = tf.float32)
    def zero():return weights[0]
    def one(): return weights[1]
    
    conf = tf.constant(conf)
    @tf.function
    def f(x):
        return tf.cond(x, zero, one)
    predclass = tf.map_fn(fn = f, elems = tf.less(tf.squeeze(y_pred), conf), fn_output_signature=tf.float32, parallel_iterations=5)
    mask = tf.where(tf.math.equal(y_true, tf.constant(0.)), predclass[:,0], predclass[:,1])

    return tf.keras.losses.binary_crossentropy(y_pred, y_true) * mask

@tf.function
def weightedceclasses(y_true, y_pred, weights, conf = 0.5):    
    conf = tf.constant(conf)
    predclass = tf.math.rint(y_pred)
    predclass = predclass * 2
    predclass = tf.add(y_true, predclass)
    d = {0: weights[0][0], 1: weights[0][1], 2: weights[1][0], 3:weights[1][1]}
    def f(x):
        print(x)
        return d[x.ref()]

    mask = tf.map_fn(f, predclass)

    return tf.keras.losses.binary_crossentropy(y_pred, y_true) * mask

@tf.function
def weightedcedef(y_true, y_pred, weights, conf = 0.5):
    def zero():return weights[0]
    def one(): return weights[1]
    
    conf = tf.constant(conf)
    
    def f(x):
        return tf.cond(x, zero, one)
    predclass = tf.map_fn(fn = f, elems = tf.less(tf.squeeze(y_pred), conf), fn_output_signature=tf.float32)
    mask = tf.where(tf.math.equal(y_true, tf.constant(0.)), predclass[:,0], predclass[:,1])

    return tf.keras.losses.binary_crossentropy(y_pred, y_true) * mask

@tf.function
def p0(y_true, y_pred, weights, conf = 0.5):
    sq = tf.less(tf.squeeze(y_pred), conf)
    return sq

@tf.function
def p1(y_true, sq, weights, conf = 0.5):
    def zero():return weights[0]
    def one(): return weights[1]
    
    conf = tf.constant(conf)
    
    def f(x):
        return tf.cond(x, zero, one)
    predclass = tf.map_fn(fn = f, elems = sq, fn_output_signature=tf.float32)
    return predclass

@tf.function
def p11(y_true, sq, weights, conf = 0.5):    
    conf = tf.constant(conf)
    
    predclass = tf.where(sq, tf.constant(0,dtype=tf.int32), tf.constant(1,dtype=tf.int32))
    predclass = tf.gather(weights, predclass)
    return predclass

@tf.function
def p2(y_true, predclass):
    return tf.where(tf.math.equal(tf.squeeze(y_true), tf.constant(0.)), predclass[:,0], predclass[:,1])

@tf.function
def p3(y_pred, y_true, mask):
    return tf.math.multiply(tf.keras.losses.binary_crossentropy(y_true, y_pred), mask)

def overallspeed():
    tf.random.set_seed(0)
    x = tf.random.uniform([1000], 0, 50)
    y = tf.random.uniform([1000], 0, 50)
    weights = np.array([[1,10],[1,1]]).astype('float32')
    times = 1

    start = time.perf_counter()
    for i in range(times):
        weightedceeager(y,x,weights)
    print(f'Eager took {time.perf_counter() - start} seconds')

    start = time.perf_counter()
    for i in range(times):
        weightedcedef(y,x,weights)
    print(f'Default took {time.perf_counter() - start} seconds')

    start = time.perf_counter()
    for i in range(times):
        weightedce10(y,x,weights)
    print(f'10 non-eager took {time.perf_counter() - start} seconds')

    start = time.perf_counter()
    for i in range(times):
        weightedce1(y,x,weights)
    print(f'1 non-eager took {time.perf_counter() - start} seconds')

    start = time.perf_counter()
    for i in range(times):
        weightedce5(y,x,weights)
    print(f'5 non-eager took {time.perf_counter() - start} seconds')

    start = time.perf_counter()
    for i in range(times):
        weightedceclasses(y,x,weights)
    print(f'5 non-eager took {time.perf_counter() - start} seconds')

if __name__ == "__main__":
    import numpy as np
    tf.random.set_seed(0)
    x = tf.random.uniform([10], 0, 1)
    y = np.concatenate([np.zeros(5), np.ones(5)]).astype('float32')
    weights = np.array([[1,1],[1,1]]).astype('float32')
    x,y = np.expand_dims(x,axis=-1), np.expand_dims(y,axis=-1)
    times = 1
    args = [y,x,weights]
    predclass, mask, sq = None, None, None
    y_true, y_pred = y,x


    start = time.perf_counter()
    for i in range(times):
        sq = p0(y_true, y_pred, weights, conf = 0.5)
    print(f'p0 {time.perf_counter() - start} seconds')
    print(sq)
    start = time.perf_counter()
    for i in range(times):
        predclass = p1(y_true, sq, weights, conf = 0.5)
    print(f'p1 {time.perf_counter() - start} seconds')
    
    start = time.perf_counter()
    for i in range(times):
        predclass = p11(y_true, sq, weights, conf = 0.5)
    print(f'p11 {time.perf_counter() - start} seconds')

    start = time.perf_counter()
    for i in range(times):
        mask = p2(y_true, predclass)
    print(f'p2 {time.perf_counter() - start} seconds')
    print(x,y, mask)
    start = time.perf_counter()
    for i in range(times):
        res = p3(y_pred, y_true, mask)
    print(f'p3 {time.perf_counter() - start} seconds')
    print('res')
    print(res, tf.keras.losses.binary_crossentropy(y,x))
    start = time.perf_counter()
    for i in range(times):
        weightedcedef(y,x,weights)
    print(f'Default took {time.perf_counter() - start} seconds')  
