

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def initialize(self):
        pass

    def forward(self, input_vector):
        pass

    def backward(self, output_error, learning_rate):
        pass

    def size(self):
        pass


class FullyConnected(Layer):
    def __init__(self, size):
        self.bias_count = size
        self.biases = None
        self.weights = None

    def initialize(self, *, weight_initializer=0, bias_initializer=0):
        self.biases = [bias_initializer] * self.bias_count

        weights_per_bias_count = self.input.size()

        self.weights = []

        for j in range(self.bias_count):
            weights_j = [weight_initializer] * weights_per_bias_count
            self.weights.append(weights_j)

    def forward(self, input_vector):
        output = []

        for bias_j in self.biases:
            output_j = bias_j
            for weight_ij, x_i in zip(self.weights, input_vector):
                output_j += weight_ij * x_i

            output.append(output_j)

        return output

    def backward(self, output_error, learning_rate):
        pass

    def size(self):
        return size


