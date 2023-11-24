# Attempting to implement https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65

import math
import random
import struct
import sys


TYPE_FORMAT = 'f'
PARAMETER_SIZE_BYTES = struct.calcsize(TYPE_FORMAT)


class Layer:
    def size(self):
        raise NotImplementedError

    def initialize(self, buffer):
        raise NotImplementedError

    def forward(self, input_matrix):
        raise NotImplementedError

    def backward(self, last_input_matrix, output_error_matrix):
        raise NotImplementedError

    def update(self, config, update_data):
        raise NotImplementedError


def dot_product(a, b):
    result = 0
    for a_i, b_i in zip(a, b):
        result += a_i * b_i
    return result


# def matrix_multiply(a, b):
#     # rows are first dimension, columns second dimension
#     result = []
#     for row in a:

#         for column in b:






class FullyConnected(Layer):
    def __init__(self, input_count, output_count):
        self.input_count = input_count
        self.output_count = output_count

    def size(self):
        # One weight for each input/output pair
        weights_size = self.input_count * self.output_count
        # One bias for each output
        biases_size = self.output_count
        return weights_size + biases_size

    def initialize(self, buffer):
        output_bytes = self.output_count * PARAMETER_SIZE_BYTES
        self.biases = buffer[:output_bytes].cast(
            TYPE_FORMAT, shape=(self.output_count,))
        for j in range(self.output_count):
            bias_j = random.normalvariate(mu=0, sigma=1)
            self.biases[j] = bias_j

        # Rows are number of inputs
        # Columns are number of outputs
        # Column vectors are the weights for one output
        self.weights = buffer[output_bytes:].cast(
            TYPE_FORMAT, shape=(self.input_count, self.output_count))

        for i in range(self.input_count):
            for j in range(self.output_count):
                weight_ij = random.normalvariate(mu=0, sigma=1)
                self.weights[i, j] = weight_ij

    def forward(self, input_matrix):
        # Row vectors are the inputs for one batch
        # Each column within a row is an input value for that specific example
        batch_size = len(input_matrix)

        # TODO: Consider using a pre-allocated array for this hidden state
        # for each iteration instead of reallocating it every time through.
        # Or possibly pass in the hidden state vector from the outside so
        # the memory buffer can be centrally manged and synchronized. Needs
        # to allow multiple forward() calls at the same time from different
        # threads/processes using the network parameters as shared memory.
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
                    weight_ij = self.weights[i, j]
                    output_j += x_i * weight_ij

                output_b.append(output_j)

            result.append(output_b)

        return result

    def backward(self, last_input_matrix, output_error_matrix):
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
                weight_ij = self.weights[i, j]

                for batch_index in range(batch_size):
                    output_error_bj = output_error_matrix[batch_index][j]
                    input_error[batch_index][i] += output_error_bj * weight_ij

        # print(f'{self.__class__.__name__}{id(self)}: InputError={input_error}')

        update_data = bias_error, weights_error
        return input_error, update_data
        # XXX split these updates up

    def update(self, config, update_data):
        bias_error, weights_error = update_data

        # Update the biases
        for j in range(self.output_count):
            bias_error_j = bias_error[j]
            self.biases[j] -= config.learning_rate * bias_error_j

        # Update weights
        for i in range(self.input_count):
            for j in range(self.output_count):
                weights_error_ij = weights_error[i][j]
                delta = config.learning_rate * weights_error_ij
                self.weights[i, j] -= delta

    # def __repr__(self):
    #     return (
    #         f'{self.__class__.__name__}('
    #         f'biases={self.biases}, '
    #         f'weights={self.weights})')


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

    def size(self):
        return 0

    def initialize(self, buffer):
        pass

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

    def backward(self, last_input_matrix, output_error_matrix):
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

        update_data = None
        return result, update_data

    def update(self, config, update_data):
        pass

    # def __repr__(self):
    #     return f'{self.__class__.__name__}()'


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


def initialize_network(network):
    total_size = 0
    for layer in network.layers:
        total_size += layer.size()


    total_bytes = PARAMETER_SIZE_BYTES * total_size
    buffer = memoryview(bytearray(total_bytes))

    offset_bytes = 0
    for layer in network.layers:
        next_size = layer.size()
        next_bytes = PARAMETER_SIZE_BYTES * next_size
        next_buffer = buffer[offset_bytes:offset_bytes+next_bytes]
        layer.initialize(next_buffer)
        offset_bytes += next_bytes


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
    input_matrix, expected_output = zip(*shard)

    # Why even have shards within a batch? Why not just run one example in each
    # shard? The reason is that you might be able to utilize in-core
    # parallelism, such as SIMD instructions, if you process many inputs
    # at the same time on the same core (e.g., using NumPy).
    history, output = feed_forward(network, input_matrix)

    mse = config.loss(expected_output, output)
    output_error = config.loss_derivative(expected_output, output)
    # print(f'LossDerivative={output_error}')

    all_updates = []

    items = list(zip(range(len(network.layers)), network.layers, history))

    for layer_index, layer, last_input in reversed(items):
        input_error, update_data = layer.backward(last_input, output_error)
        output_error = input_error
        all_updates.append((layer_index, update_data))

    return mse, all_updates


def train_batch(network, config, batch):
    error_sum = 0
    error_count = 0
    all_updates = []

    # This step is parallelizable, where each shard can be computed on a
    # separate thread / core / machine and the update gradients are
    # retrieved for that portion of the batch.
    for shard in batch:
        mse, updates = train_shard(network, config, shard)
        error_sum += sum(mse)
        error_count += len(shard)
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


def train(network, config, examples):
    print('Start training')
    # for i, layer in enumerate(network.layers, 1):
        # print(f'Layer {i}: {layer}')

    for epoch_index in range(config.epochs):
        random.shuffle(examples)

        shard_size = config.batch_size // config.parallelism
        shard_it = iter_grouped(examples, shard_size)
        batch_it = iter_grouped(shard_it, config.parallelism)

        total_error_sum = 0
        total_error_count = 0

        for batch in batch_it:
            error_sum, error_count = train_batch(network, config, batch)
            total_error_sum += error_sum
            total_error_count += error_count
            avg_error = total_error_sum / float(total_error_count)
            print(
                f'Epoch={epoch_index+1}, '
                f'Examples={total_error_count}, '
                f'AvgError={avg_error:.10f}')


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


"""
TODO
- Allow for resuming training from a previous network
- Allow control-C to interrupt the training and save a clean snapshot; need
  to run the training on another thread to make this work right.
- Compare performance:
    - Use batch-size-efficient generated matmul functions for inner loops
    - Use C-extension matmul functions for inner loops
    - Use numpy matmul functions for inner loops
- Parallelize feed forward for each item in a batch across multiple processes
  using multiprocessing, then apply backpropagation in a single process
  - Allocate the network in a shared memory buffer for all parameters
    (like weights and biases):
    https://docs.python.org/3/library/multiprocessing.shared_memory.html
  - Use https://docs.python.org/3/library/stdtypes.html#memoryview to treat
    the parameters as basic arrays that can be used in forward/backward

"""
