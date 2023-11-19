# Attempting to implement https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65

import random


class Layer:
    def __init__(self):
        self.last_input = None

    def forward(self, input_vector):
        pass

    def backward(self, output_error, learning_rate):
        pass


class FullyConnected(Layer):
    def __init__(self, input_count, output_count):
        super().__init__()
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
        self.last_input = input_vector

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

    def backward(self, output_error, learning_rate):
        # Output error is ∂E/∂Y
        bias_error = output_error

        # Same size as the weights matrix
        # Rows are number of inputs
        # Columns are number of outputs
        weights_error = []
        for i in range(self.input_count):
            input_i = self.last_input[i]
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

    def backward(self, output_error, learning_rate):
        input_error = []
        for i in range(self.count):
            input_i = self.last_input[i]
            output_error_i = output_error[i]
            output_i_derivative = self.function_derivative(input_i)
            input_error_i = output_i_derivative * output_error_i
            input_error.append(input_error_i)

        return input_error
