import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from scipy.io import loadmat
import matplotlib.pyplot as plt
from models import VAEdistance


def make_federated_data(client_data, client_ids):
    return [
        preprocess(client_data.create_tf_dataset_for_client(x))
        for x in client_ids
    ]

def createVAE(inputs, model):
    layersizes = np.array([2**w for w in range(2,12)])
    layersizes = layersizes[layersizes < inputs.shape[1]]

    if len(layersizes) > 5:
        layersizes = layersizes[::2]

    layersizes = list(reversed(layersizes))

    return model(inputsize = [], inlayersize = layersizes,
            outputsize = [[inputs.shape[1]]],latentsize = 4, finalactivation = ['linear'])

class Ignore(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def update_state(self, y_true, y_pred, sample_weight=None):
        return

    def result(self):
        return 0.
    
    def reset_states(self):
        return

    def call(self, y_true, y_pred):
        return 0.

if __name__ == '__main__':
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
    mat = loadmat('datasets/mnist.mat')
    NUM_CLIENTS = 20
    NUM_EPOCHS = 20
    BATCH_SIZE = 100
    SHUFFLE_BUFFER = 100
    PREFETCH_BUFFER = 10

    example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])

    tf.random.set_seed(0)

    def preprocess(dataset):

        def batch_format_fn(element):
            """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
            x=tf.reshape(element['pixels'], [-1, 784])
            y=tf.reshape(element['label'], [-1, 1])
            ind = tf.squeeze(tf.math.logical_or(y == 0, y == 6), -1)
            y = tf.where(y[ind] == 6, 1, 0)
            return collections.OrderedDict(x = x[ind], y = tf.convert_to_tensor([y,y]))

        return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
            BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

    preprocessed_example_dataset = preprocess(example_dataset)

    sample_batch = tf.nest.map_structure(lambda x: x.numpy(),
                                        next(iter(preprocessed_example_dataset)))

    def model_fn(inputs, model):
        # We _must_ create a new model here, and _not_ capture it from an external
        # scope. TFF will call this within different graph contexts.
        keras_model = createVAE(inputs, model)
        keras_model(inputs)
        keras_model.outputs = [0,1]
        return tff.learning.from_keras_model(
            keras_model,
            input_spec=preprocessed_example_dataset.element_spec,
            loss=[tf.keras.losses.MeanAbsoluteError(),Ignore()])

    sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

    federated_train_data = make_federated_data(emnist_train, sample_clients)

    print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
    print('First dataset: {d}'.format(d=federated_train_data[0]))

    iterative_process = tff.learning.build_federated_averaging_process(
    lambda: model_fn(sample_batch['x'], VAEdistance),
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

    state = iterative_process.initialize()
    state, metrics = iterative_process.next(state, federated_train_data)
    print('round  1, metrics={}'.format(metrics))
    NUM_ROUNDS = 11
    for round_num in range(2, NUM_ROUNDS):
        state, metrics = iterative_process.next(state, federated_train_data)
        print('round {:2d}, metrics={}'.format(round_num, metrics))
    
    # Copy weights to a 'normal' non tff model and predict
    model_for_inference = createVAE(sample_batch['x'], VAEdistance)
    model_for_inference(sample_batch['x'])
    state.model.assign_weights_to(model_for_inference)

    test = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[400])
    test = preprocess(test)
    test = tf.nest.map_structure(lambda x: x.numpy(),
                                        next(iter(test)))
    predictions = np.mean(model_for_inference.predict(test['x'])[0],0)
    print(test['x'].shape)
    print(predictions)