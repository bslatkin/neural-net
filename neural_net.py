# Attempting to implement https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65

import random


class Layer:
    def forward(self, input_vector):
        pass

    def backward(self, output_error, learning_rate):
        pass


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
        self.weights = []

        for i in range(self.input_count):
            weights_i = []

            for j in range(self.output_count):
                weight_ij = random.uniform(-0.5, 0.5)
                weights_i.append(weight_ij)

            self.weights.append(weights_i)

    def forward(self, input_vector):
        output = []

        for j in range(self.output_count):
            bias_j = self.biases[j]
            output_j = bias_j

            for i in range(self.input_count):
                weight_ij = self.weights[i][j]
                x_i = input_vector[i]
                output_j += weight_ij * x_i

            output.append(output_j)

        return output

    def backward(self, last_input, output_error, learning_rate):
        # Output error is ∂E/∂Y
        bias_error = output_error

        # Same size as the weights matrix
        # Rows are number of inputs
        # Columns are number of outputs
        weights_error = []
        for i in range(self.input_count):
            input_i = last_input[i]
            weights_error_i = []

            for j in range(self.output_count):
                output_error_j = output_error[j]
                weight_error_ij = output_error_j * input_i
                weights_error_i.append(weight_error_ij)

            weights_error.append(weights_error_i)

        # Same size as the input
        input_error = []
        for i in range(self.input_count):
            input_error_i = 0

            for j in range(self.output_count):
                output_error_j = output_error[j]
                weight_ij = self.weights[i][j]
                input_error_i += output_error_j * weight_ij

            input_error.append(input_error_i)

        # Update the biases
        for j in range(self.output_count):
            bias_error_j = bias_error[j]
            self.biases[j] -= learning_rate * bias_error_j

        # Update weights
        for i in range(self.input_count):
            for j in range(self.output_count):
                weights_error_ij = weights_error[i][j]
                self.weights[i][j] -= learning_rate * weights_error_ij

        return input_error

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'biases={self.biases}, '
            f'weights={self.weights})')


def relu(x):
    if x >= 0:
        return x
    else:
        return 0


def relu_derivative(x):
    if x >= 0:
        return 1
    else:
        return 0


class Activation(Layer):
    def __init__(self, count, function, function_derivative):
        super().__init__()
        self.count = count
        self.function = function
        self.function_derivative = function_derivative

    def forward(self, input_vector):
        self.last_input = input_vector

        output = []
        for i in range(self.count):
            input_i = input_vector[i]
            output_i = self.function(input_i)
            output.append(output_i)

        return output

    def backward(self, last_input, output_error, learning_rate):
        input_error = []
        for i in range(self.count):
            input_i = last_input[i]
            output_error_i = output_error[i]
            output_i_derivative = self.function_derivative(input_i)
            input_error_i = output_i_derivative * output_error_i
            input_error.append(input_error_i)

        return input_error

    def __repr__(self):
        return f'{self.__class__.__name__}()'


def mean_squared_error(desired, found):
    total = 0
    for desired_i, found_i in zip(desired, found):
        delta = desired_i - found_i
        total += delta ** 2
    mean = total / len(desired)
    return mean


def mean_squared_error_derivative(desired, found):
    result = []
    for desired_i, found_i in zip(desired, found):
        delta = found_i - desired_i
        scaled = 2 / len(desired) * delta
        result.append(scaled)
    return result



# add an l2


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
                 learning_rate):
        self.loss = loss
        self.loss_derivative = loss_derivative
        self.epochs = epochs
        self.learning_rate = learning_rate


def feed_forward(network, input_vector):
    next_input = input_vector
    history = []

    for layer in network.layers:
        history.append(next_input)
        output = layer.forward(next_input)
        next_input = output

    return history, output


def train_one(network, config, input_vector, expected_output):
    history, output = feed_forward(network, input_vector)

    # TODO: For batching, the output error gradient would be averaged
    # across all of the samples in the batch.
    mse = config.loss(expected_output, output)
    output_error = config.loss_derivative(expected_output, output)
    print(f'LossDerivative={output_error}')

    for layer, last_input in zip(reversed(network.layers), reversed(history)):
        output_error = layer.backward(
            last_input, output_error, config.learning_rate)

    return mse


def train(network, config, examples):
    for epoch_index in range(config.epochs):
        error_sum = 0
        error_count = 0

        for input_vector, expected_output in examples:
            mse = train_one(network, config, input_vector, expected_output)
            error_sum += mse
            error_count += 1

            print(
                f'Epoch={epoch_index+1}, '
                f'Example={error_count}, '
                f'AvgError={error_sum/error_count:.10f}')

            for i, layer in enumerate(network.layers, 1):
                print(f'Layer {i}: {layer}')


def predict(network, input_vector):
    _, output = feed_forward(network, input_vector)
    return output


def test_xor():
    network = Network()
    network.add(FullyConnected(2, 3))
    network.add(Activation(3, relu, relu_derivative))
    network.add(FullyConnected(3, 1))
    network.add(Activation(1, relu, relu_derivative))

    config = TrainingConfig(
        loss=mean_squared_error,
        loss_derivative=mean_squared_error_derivative,
        epochs=1000,
        learning_rate=0.1)

    labeled_examples = [
        ((0, 0), (0,)),
        ((0, 1), (1,)),
        ((1, 0), (1,)),
        ((1, 1), (0,)),
    ]

    train(network, config, labeled_examples)

    test_examples = labeled_examples

    error_sum = 0
    error_count = 0

    for input_vector, expected_output in test_examples:
        output = predict(network, input_vector)
        print(f'Input={input_vector}, Output={output}')

        mse = config.loss(expected_output, output)
        error_sum += mse
        error_count += 1

    print(f'AvgError={error_sum/error_count:.10f}')


test_xor()
