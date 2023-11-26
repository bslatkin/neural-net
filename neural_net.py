# Attempting to implement https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65

import math
import random
import struct
import sys


# Prevent NumPy from using multiple threads per process, which can cause
# the multi-processing version of this program to be slower than the single
# process version. This environment variable must be set before numpy is
# imported. More info here:
# https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy#comment87232711_31622299
import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np


class Layer:
    def initialize(self):
        raise NotImplementedError

    def forward(self, input_matrix):
        raise NotImplementedError

    def backward(self, last_input_matrix, output_error_matrix):
        raise NotImplementedError

    def update(self, config, update_data):
        raise NotImplementedError


# XXX first I need a better matrix class
# class Optimizer:
#     def __call__(self, gradient_matrix):
#         pass


# class VectorOptimizer:
#     def __init__(self, count, learning_rate):
#         self.count = count
#         self.learning_rate = learning_rate

#     def __call__(self, gradient_vector):
#         result = []
#         for element in gradient_vector:
#             adjusted = self.learning_rate * element
#             result.append(adjusted)
#         return result


class FullyConnected(Layer):
    def __init__(self, input_count, output_count):
        self.input_count = input_count
        self.output_count = output_count

        self.biases = np.zeros(
            shape=(self.output_count,),
            dtype=np.float64)

        self.weights = np.zeros(
            shape=(self.input_count, self.output_count),
            dtype=np.float64)

    def initialize(self):
        for j in range(self.output_count):
            bias_j = random.normalvariate(mu=0, sigma=1)
            self.biases[j] = bias_j

        # Rows are number of inputs
        # Columns are number of outputs
        # Column vectors are the weights for one output
        for i in range(self.input_count):
            for j in range(self.output_count):
                weight_ij = random.normalvariate(mu=0, sigma=1)
                self.weights[i, j] = weight_ij

    def forward(self, input_matrix):
        # Row vectors are the inputs for one example
        # Each column within a row is an input value for that specific example
        # Compute Y = XW + B
        multiplied = input_matrix @ self.weights

        # Biases is a vector, and each value of that needs to be
        # added to the corresponding column for each row.
        added = multiplied + self.biases

        return added

    def backward(self, last_input_matrix, output_error_matrix):
        # Output error is ∂E/∂Y, each row is an example, and each column in each
        # row is the error gradient for that output position. But the biases
        # in this FullyConnected layer are merely a vector. So this sums all
        # of the gradients for each output positions across all examples
        # in order to calculate the error gradient for the bias update.
        # This took me a long time to figure out! This post helped:
        # https://stats.stackexchange.com/questions/373163/how-are-biases-updated-when-batch-size-1
        bias_error = np.sum(output_error_matrix, axis=1)

        # Same size as the weights matrix
        # Rows are number of inputs
        # Columns are number of outputs
        #
        # X^T * ∂E/∂Y
        #
        # x has one row per example with columns being input values
        # x^t has each example as a column, so each row is the i-th input across all examples
        #
        # ∂E/∂Y has one row per example, each column is the j-th output
        # the weight error at each point ij is the dot product
        # of the i-th input across all examples with the j-th output
        # across all examples.
        weights_error = last_input_matrix.T @ output_error_matrix

        # Same size as the input
        # Rows are examples
        # Columns are inputs
        # ∂E/∂Y * W^T
        input_error = output_error_matrix @ self.weights.T

        # XXX work towards this
        # adjusted_bias_error = self.bias_optimizer(bias_error)
        # adjusted_weights_error = self.weights_optimizer(weights_error)

        update_data = bias_error, weights_error
        return input_error, update_data

    def update(self, config, update_data):
        bias_error, weights_error = update_data

        # Update the biases
        bias_delta = config.learning_rate * bias_error
        self.biases -= np.sum(bias_delta, axis=0)

        # Update weights
        weights_delta = config.learning_rate * weights_error
        self.weights -= weights_delta


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


def sigmoid_derivative(value):
    f_x = sigmoid(value)
    return f_x * (1 - f_x)


class Activation(Layer):
    def __init__(self, count, function, function_derivative):
        self.count = count
        self.function = np.vectorize(function)
        self.function_derivative = np.vectorize(function_derivative)

    def parameters_count(self):
        return 0

    def initialize(self):
        pass

    def forward(self, input_matrix):
        # Output exactly matches the input
        # Row contains all inputs for one example
        # Each column the input at that position for the one example
        return self.function(input_matrix)

    def backward(self, last_input_matrix, output_error_matrix):
        # Calculating ∂E/∂Y (elementwise multiply) ∂Y/∂X to get ∂E/∂X
        input_gradient = self.function_derivative(last_input_matrix)
        result = output_error_matrix * input_gradient
        update_data = None
        return result, update_data

    def update(self, config, update_data):
        pass


def mean_squared_error(desired_matrix, found_matrix):
    # For found matrix and desired matrix
    # Row vectors are examples
    # Columns are outputs for a specific example
    delta_matrix = np.subtract(desired_matrix, found_matrix)
    squared_matrix = np.power(delta_matrix, 2)
    summed_matrix = np.sum(squared_matrix, axis=1)
    average_matrix = summed_matrix / delta_matrix.shape[1]
    return average_matrix


def mean_squared_error_derivative(desired_matrix, found_matrix):
    # Row vectors are examples
    # Columns are outputs for a specific example
    delta_matrix = np.subtract(found_matrix, desired_matrix)
    # Each result row contains gradients for each column of an example
    result = 2 / delta_matrix.shape[1] * delta_matrix
    return result


class Network:
    def __init__(self, type_format='f'):
        self.type_format = type_format
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def initialize(self):
        for layer in self.layers:
            layer.initialize()


class TrainingConfig:
    def __init__(self,
                 *,
                 loss,
                 loss_derivative,
                 epochs,
                 batch_size,
                 parallelism,
                 learning_rate):
        self.loss = loss
        self.loss_derivative = loss_derivative
        self.epochs = epochs
        self.batch_size = batch_size
        self.parallelism = parallelism
        self.learning_rate = learning_rate


def feed_forward(network, input_matrix):
    next_input = input_matrix
    history = []

    for layer in network.layers:
        history.append(next_input)
        output = layer.forward(next_input)
        next_input = output

    return history, output


def train_shard(network, config, shard):
    input_examples, expected_output = zip(*shard)

    # TODO: Move this into example creation to avoid having to do the
    # conversion repeatedly for the same input examples.
    input_matrix = np.array(input_examples)
    expected_matrix = np.array(expected_output)

    # Why even have shards within a batch? Why not just run one example in each
    # shard? The reason is that you might be able to utilize in-core
    # parallelism, such as SIMD instructions, if you process many inputs
    # at the same time on the same core (e.g., using NumPy).
    history, output = feed_forward(network, input_matrix)

    mse = config.loss(expected_matrix, output)
    output_error = config.loss_derivative(expected_matrix, output)

    all_updates = []

    items = list(zip(range(len(network.layers)), network.layers, history))

    for layer_index, layer, last_input in reversed(items):
        input_error, update_data = layer.backward(last_input, output_error)
        output_error = input_error
        if update_data:
            all_updates.append((layer_index, update_data))

    return mse, all_updates


def train_batch(network, config, executor, batch):
    error_sum = 0
    error_count = 0
    all_updates = []

    # This step is parallelizable, where each shard can be computed on a
    # separate thread / core / machine and the update gradients are
    # retrieved for that portion of the batch.
    # for shard in batch:
    all_futures = []
    for shard in batch:
        future = executor.submit(train_shard, network, config, shard)
        all_futures.append(future)

    for future in all_futures:
        mse, updates = future.result()
        error_sum += sum(mse)
        error_count += len(mse)
        all_updates.extend(updates)

    # This is the "all to all" step, where the model weights and biases for
    # each layer are updated based on the error gradients computed in parallel
    # for the step above.
    for layer_index, update_data in all_updates:
        network.layers[layer_index].update(config, update_data)

    return error_sum, error_count


def iter_grouped(items, group_size):
    it = iter(items)

    while True:
        next_slice = []
        for _ in range(group_size):
            try:
                next_item = next(it)
            except StopIteration:
                if next_slice:
                    yield next_slice

                return
            else:
                next_slice.append(next_item)

        yield next_slice


def train(network, config, executor, examples):
    print('Start training')

    for epoch_index in range(config.epochs):
        # TODO: Do all of the data shuffling and batching before training
        # so these example tensor batches are all dense and don't need to be
        # repeatedly recreated for each epoch.
        random.shuffle(examples)

        shard_size = max(1, config.batch_size // config.parallelism)
        shard_it = iter_grouped(examples, shard_size)
        batch_it = iter_grouped(shard_it, config.parallelism)

        total_examples_count = 0

        for batch in batch_it:
            error_sum, examples_count = train_batch(
                network, config, executor, batch)

            total_examples_count += examples_count
            avg_error = error_sum / float(examples_count)

            print(
                f'Epoch={epoch_index+1}, '
                f'Examples={total_examples_count}, '
                f'AvgError={avg_error:.10f}')


def predict(network, input_vector):
    # TODO: Move this into example creation to avoid having to do the
    # conversion repeatedly for the same input examples.
    input_matrix = np.array([input_vector])
    _, output = feed_forward(network, input_matrix)
    return output


def profile_func(func, *args, **kwargs):
    import cProfile
    import pstats
    from pstats import SortKey

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        return func(*args, **kwargs)
    finally:
        profiler.disable()

        stats = pstats.Stats(profiler, stream=sys.stderr)
        stats = stats.strip_dirs()
        stats = stats.sort_stats(SortKey.CUMULATIVE, SortKey.TIME, SortKey.NAME)

        stats.print_stats()
        stats.print_callers()


"""
TODO
- Add an Adam optimizer to adjust learning rate over time
- Compare performance:
    - Use batch-size-efficient generated matmul functions for inner loops
    - Use C-extension matmul functions for inner loops
    - Use numpy matmul functions for inner loops
"""


