# Attempting to implement https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65

import math
import random
import sys


class Layer:
    def forward(self, input_matrix):
        raise NotImplementedError

    def backward(self, output_error_matrix, learning_rate):
        raise NotImplementedError


def dot_product(a, b):
    result = 0
    for a_i, b_i in zip(a, b):
        result += a_i * b_i
    return result


class FullyConnected(Layer):
    def __init__(self, input_count, output_count):
        self.input_count = input_count
        self.output_count = output_count

        # One bias for each output
        self.biases = []
        for j in range(self.output_count):
            bias_j = random.uniform(-0.5, 0.5)
            self.biases.append(bias_j)

        # Rows are number of inputs
        # Columns are number of outputs
        # Column vectors are the weights for one output
        self.weights = []

        for i in range(self.input_count):
            weights_i = []

            for j in range(self.output_count):
                weight_ij = random.uniform(-0.5, 0.5)
                weights_i.append(weight_ij)

            self.weights.append(weights_i)

    # @profile
    def forward(self, input_matrix):
        # Row vectors are the inputs for one batch
        # Each column is an input value for that specific example
        batch_size = len(input_matrix)
        input_count = len(input_matrix[0])

        assert input_count == self.input_count, \
            f'{input_count=} == {self.input_count=}'

        result = []

        # Output has batch size rows, output size columns
        for b in range(batch_size):
            input_b = input_matrix[b]
            output_b = []

            for j in range(self.output_count):
                bias_j = self.biases[j]
                output_j = bias_j

                for i in range(self.input_count):
                    x_i = input_b[i]
                    weight_ij = self.weights[i][j]
                    output_j += x_i * weight_ij

                output_b.append(output_j)

            result.append(output_b)

        return result

    # @profile
    def backward(self, last_input_matrix, output_error_matrix, config):
        # print(f'{self.__class__.__name__}{id(self)}: OutputError={output_error}')

        batch_size = len(output_error_matrix)

        # Output error is ∂E/∂Y, each row is a batch, and each column in each
        # row is the error gradient for that output position. But the biases
        # in this FullyConnected layer are merely a vector. So this sums all
        # of the gradients for each output positions across all batches
        # in order to calculate the error gradient for the bias update.
        # This took me a long time to figure out! This post helped:
        # https://stats.stackexchange.com/questions/373163/how-are-biases-updated-when-batch-size-1
        bias_error = []
        for j in range(self.output_count):
            bias_error_j = 0
            for i in range(batch_size):
                output_error_ij = output_error_matrix[i][j]
                bias_error_j += output_error_ij
            bias_error.append(bias_error_j)

        # Same size as the weights matrix
        # Rows are number of inputs
        # Columns are number of outputs
        # X^T * ∂E/∂Y
        weights_error = []
        for i in range(self.input_count):
            weights_i = []
            for j in range(self.output_count):
                weights_i.append(0)
            weights_error.append(weights_i)

        # x has one row per batch with columns being input values
        # x^t has each batch as a column, so each row is the i-th input across all batches

        # ∂E/∂Y has one row per batch, each column is the j-th output
        # the weight error at each point ij is the dot product
        # of the i-th input across all batches with the j-th output
        # across all batches.
        for i in range(self.input_count):
            for j in range(self.output_count):
                delta = 0

                for batch_index in range(batch_size):
                    input_bi = last_input_matrix[batch_index][i]
                    output_error_bj = output_error_matrix[batch_index][j]
                    delta += input_bi * output_error_bj

                weights_error[i][j] = delta

        # print(f'{self.__class__.__name__}{id(self)}: WeightsError={weights_error}')

        # Same size as the input
        # Rows are batches
        # Columns are inputs
        # ∂E/∂Y * W^T
        input_error = []
        for batch_index in range(batch_size):
            input_error_b = []
            for i in range(self.input_count):
                input_error_b.append(0)
            input_error.append(input_error_b)

        for i in range(self.input_count):
            for j in range(self.output_count):
                weight_ij = self.weights[i][j]

                for batch_index in range(batch_size):
                    output_error_bj = output_error_matrix[batch_index][j]
                    input_error[batch_index][i] += output_error_bj * weight_ij

        # print(f'{self.__class__.__name__}{id(self)}: InputError={input_error}')

        # Update the biases
        for j in range(self.output_count):
            bias_error_j = bias_error[j]
            self.biases[j] -= config.learning_rate * bias_error_j

        # Update weights
        for i in range(self.input_count):
            for j in range(self.output_count):
                weights_error_ij = weights_error[i][j]
                delta = config.learning_rate * weights_error_ij
                self.weights[i][j] -= delta

        return input_error

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'biases={self.biases}, '
            f'weights={self.weights})')


def sigmoid(input_vector):
    result = []
    for item in input_vector:
        value = 1 / (1 + math.exp(-item))
        result.append(value)
    return result


def sigmoid_derivative(input_vector):
    result = []
    for f_x in sigmoid(input_vector):
        value = f_x * (1 - f_x)
        result.append(value)
    return result


class Activation(Layer):
    def __init__(self, count, function, function_derivative):
        super().__init__()
        self.count = count
        self.function = function
        self.function_derivative = function_derivative

    # @profile
    def forward(self, input_matrix):
        batch_size = len(input_matrix)

        result = []

        for batch_index in range(batch_size):
            if batch_index >= len(input_matrix):
                breakpoint()
            input_b = input_matrix[batch_index]
            output_b = self.function(input_b)
            result.append(output_b)

        return result

    # @profile
    def backward(self, last_input_matrix, output_error_matrix, learning_rate):
        # print(f'{self.__class__.__name__}{id(self)}: OutputError={output_error}')

        batch_size = len(last_input_matrix)

        result = []

        for batch_index in range(batch_size):
            last_input_b = last_input_matrix[batch_index]
            output_error = output_error_matrix[batch_index]
            output_derivative = self.function_derivative(last_input_b)

            input_error_b = []

            for error_i, derivative_i in zip(output_error, output_derivative):
                input_error_i = error_i * derivative_i
                input_error_b.append(input_error_i)

            result.append(input_error_b)

        return result

    def __repr__(self):
        return f'{self.__class__.__name__}()'


def mean_squared_error(desired_matrix, found_matrix):
    batch_size = len(desired_matrix)

    result = []

    for batch_index in range(batch_size):
        desired = desired_matrix[batch_index]
        found = found_matrix[batch_index]

        total = 0
        for desired_i, found_i in zip(desired, found):
            delta = desired_i - found_i
            total += delta ** 2
        mean = total / len(desired)

        result.append(mean)

    return result


def mean_squared_error_derivative(desired_matrix, found_matrix):
    # Row vectors are batches
    # Columns are outputs for a specific batch
    batch_size = len(desired_matrix)

    result = []

    for batch_index in range(batch_size):
        desired = desired_matrix[batch_index]
        found = found_matrix[batch_index]

        # Each result row contains gradients for each column of a batch
        batch_result = []

        for desired_i, found_i in zip(desired, found):
            delta = found_i - desired_i
            scaled = 2 / len(desired) * delta
            batch_result.append(scaled)

        result.append(batch_result)

    return result


class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)


class TrainingConfig:
    def __init__(self,
                 *,
                 loss,
                 loss_derivative,
                 epochs,
                 batch_size,
                 learning_rate):
        self.loss = loss
        self.loss_derivative = loss_derivative
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate


def feed_forward(network, input_matrix):
    next_input = input_matrix
    history = []

    for layer in network.layers:
        history.append(next_input)
        output = layer.forward(next_input)
        next_input = output

    return history, output


def train_batch(network, config, batch):
    input_matrix, expected_output = zip(*batch)

    history, output = feed_forward(network, input_matrix)

    # TODO: For batching, the output error gradient would be averaged
    # across all of the samples in the batch.
    mse = config.loss(expected_output, output)
    output_error = config.loss_derivative(expected_output, output)
    # print(f'LossDerivative={output_error}')

    for layer, last_input in zip(reversed(network.layers), reversed(history)):
        output_error = layer.backward(last_input, output_error, config)

    return mse


def iter_batch(examples, batch_size):
    for i in range(0, len(examples), batch_size):
        next_slice = examples[i:i+batch_size]
        yield next_slice


def train(network, config, examples):
    print('Start training')
    # for i, layer in enumerate(network.layers, 1):
        # print(f'Layer {i}: {layer}')

    for epoch_index in range(config.epochs):
        error_sum = 0
        error_count = 0

        # random.shuffle(examples)

        for i, batch in enumerate(iter_batch(examples, config.batch_size)):
            mse = train_batch(network, config, batch)
            error_sum += sum(mse)
            error_count += len(batch)

            print(
                f'Epoch={epoch_index+1}, '
                f'Examples={error_count}, '
                f'AvgError={error_sum/error_count:.10f}')

            # for i, layer in enumerate(network.layers, 1):
            #     print(f'Layer {i}: {layer}')


def predict(network, input_vector):
    input_matrix = [input_vector]
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
