# Attempting to implement https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65

import math
import random
import struct
import sys

from fast_math import *


class Layer:
    def parameters_count(self):
        raise NotImplementedError

    def connect(self, parameters):
        raise NotImplementedError

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
        self.parameters = None

    def __getstate__(self):
        if self.parameters:
            parameters_bytes = bytes(self.parameters.cast('B'))
            type_format = self.parameters.format
        else:
            parameters_bytes = None
            type_format = None

        return dict(
            input_count=self.input_count,
            output_count=self.output_count,
            parameters_bytes=parameters_bytes,
            type_format=type_format)

    def __setstate__(self, state):
        parameters_bytes = state.pop('parameters_bytes')
        type_format = state.pop('type_format')
        self.__dict__.update(state)

        if parameters_bytes:
            parameters = memoryview(parameters_bytes).cast(type_format)
            self.connect(parameters)

    def parameters_count(self):
        # One weight for each input/output pair
        weights_size = self.input_count * self.output_count
        # One bias for each output
        biases_size = self.output_count
        return weights_size + biases_size

    def connect(self, parameters):
        self.parameters = parameters

        self.biases = Tensor(
            1,
            self.output_count,
            data=parameters[:self.output_count])

        self.weights = Tensor(
            self.output_count,
            self.input_count,
            data=parameters[self.output_count:])

    def initialize(self):
        for j in range(self.output_count):
            bias_j = random.normalvariate(mu=0, sigma=1)
            self.biases.set(0, j, bias_j)

        # Rows are number of inputs
        # Columns are number of outputs
        # Column vectors are the weights for one output
        for j in range(self.output_count):
            for i in range(self.input_count):
                weight_ji = random.normalvariate(mu=0, sigma=1)
                self.weights.set(j, i, weight_ji)

    def forward(self, input_matrix):
        # Row vectors are the inputs for one example
        # Each column within a row is an input value for that specific example
        # Compute Y = XW + B
        multiplied = matrix_multiply(input_matrix, self.weights)

        # Biases is a column vector, and each value of that needs to be
        # added to the corresponding column for each row.
        #
        # TODO: Make this a more obvious matrix operation?
        result = Tensor(multiplied.columns, multiplied.rows)

        for j in range(result.rows):
            for i in range(result.columns):
                value = multiplied.get(i, j)
                bias = self.biases.get(0, i)
                added = value + bias
                result.set(i, j, added)

        return result

    def backward(self, last_input_matrix, output_error_matrix):
        # Output error is ∂E/∂Y, each row is an example, and each column in each
        # row is the error gradient for that output position. But the biases
        # in this FullyConnected layer are merely a vector. So this sums all
        # of the gradients for each output positions across all examples
        # in order to calculate the error gradient for the bias update.
        # This took me a long time to figure out! This post helped:
        # https://stats.stackexchange.com/questions/373163/how-are-biases-updated-when-batch-size-1
        bias_error = Tensor(self.output_count, 1)
        for j in range(self.output_count):
            bias_error_j = 0
            for i in range(output_error_matrix.rows):
                output_error_ji = output_error_matrix.get(j, i)
                bias_error_j += output_error_ji
            bias_error.set(j, 0, bias_error_j)

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
        weights_error = matrix_multiply(
            last_input_matrix.transpose(),
            output_error_matrix)

        # Same size as the input
        # Rows are examples
        # Columns are inputs
        # ∂E/∂Y * W^T
        input_error = matrix_multiply(
            output_error_matrix,
            self.weights.transpose())

        # XXX work towards this
        # adjusted_bias_error = self.bias_optimizer(bias_error)
        # adjusted_weights_error = self.weights_optimizer(weights_error)

        update_data = bias_error, weights_error
        return input_error, update_data

    def update(self, config, update_data):
        bias_error, weights_error = update_data

        # Update the biases
        for j in range(self.output_count):
            bias_error_j = bias_error.get(j, 0)
            bias_delta = config.learning_rate * bias_error_j

            bias_j = self.biases.get(0, j)
            next_bias_j = bias_j - bias_delta
            self.biases.set(0, j, next_bias_j)

        # Update weights
        for i in range(self.input_count):
            for j in range(self.output_count):
                weight_error_ji = weights_error.get(j, i)
                weight_delta = config.learning_rate * weight_error_ji

                weight_ji = self.weights.get(j, i)
                next_weight_ji = weight_ji - weight_delta
                self.weights.set(j, i, next_weight_ji)


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


def sigmoid_derivative(value):
    f_x = sigmoid(value)
    return f_x * (1 - f_x)


class Activation(Layer):
    def __init__(self, count, function, function_derivative):
        self.count = count
        self.function = function
        self.function_derivative = function_derivative

    def parameters_count(self):
        return 0

    def connect(self, parameters):
        pass

    def initialize(self):
        pass

    def forward(self, input_matrix):
        # Output exactly matches the input
        # Row contains all inputs for one example
        # Each column the input at that position for the one example
        return matrix_apply(self.function, input_matrix)

    def backward(self, last_input_matrix, output_error_matrix):
        # Calculating ∂E/∂Y (elementwise multiply) ∂Y/∂X to get ∂E/∂X
        input_gradient = matrix_apply(
            self.function_derivative, last_input_matrix)
        result = matrix_elementwise_multiply(
            output_error_matrix,
            input_gradient)
        update_data = None
        return result, update_data

    def update(self, config, update_data):
        pass


def mean_squared_error(desired_matrix, found_matrix):
    # For found matrix and desired matrix
    # Row vectors are examples
    # Columns are outputs for a specific example
    delta_matrix = matrix_elementwise_subtract(desired_matrix, found_matrix)

    squared_matrix = matrix_apply(
        lambda x: x**2,
        delta_matrix)

    # Output is a column vector with one value per example
    summed_matrix = matrix_rowwise_apply(
        sum,
        squared_matrix)

    average_matrix = matrix_apply(
        lambda total: total / delta_matrix.columns,
        summed_matrix)

    # Convert it to a list for each usage elsewhere in eval pipeline
    # TODO: Consider using a tensor everywhere
    return list(average_matrix.column(0))


def mean_squared_error_derivative(desired_matrix, found_matrix):
    # Row vectors are examples
    # Columns are outputs for a specific example
    delta_matrix = matrix_elementwise_subtract(found_matrix, desired_matrix)

    # Each result row contains gradients for each column of an example
    result = matrix_apply(
        lambda delta: 2 / delta_matrix.columns * delta,
        delta_matrix)

    return result


class Network:
    def __init__(self, type_format='f'):
        self.type_format = type_format
        self.layers = []
        self.parameters_bytes = None

    def add(self, layer):
        self.layers.append(layer)

    def parameters_count(self):
        total_count = 0
        for layer in self.layers:
            total_count += layer.parameters_count()
        return total_count

    def allocate_parameters(self):
        size_bytes = struct.calcsize(self.type_format)
        self.parameters_bytes = bytearray(size_bytes * self.parameters_count())

    def connect(self):
        parameters = memoryview(self.parameters_bytes).cast(self.type_format)
        offset = 0

        for layer in self.layers:
            next_count = layer.parameters_count()
            next_parameters = parameters[offset:offset+next_count]
            layer.connect(next_parameters)
            offset += next_count

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
    input_matrix = Tensor.from_list(input_examples)
    expected_matrix = Tensor.from_list(expected_output)

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
    input_matrix = Tensor.from_list([input_vector])
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


