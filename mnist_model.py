import struct
import sys

from neural_net import *

# Loader derived from https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook

# One-hot encoded values for the labels, 0-9
INDEX_TO_BITS = [
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
]

BITS_TO_INDEX = {}
for i, bits in enumerate(INDEX_TO_BITS):
    BITS_TO_INDEX[bits] = i


def load_mnist_data(features_path, labels_path):
    with open(features_path, 'rb') as f:
        features_data = f.read()

    with open(labels_path, 'rb') as f:
        labels_data = f.read()

    magic, labels_size = struct.unpack('>II', labels_data[:8])
    assert magic == 2049

    magic, features_size, rows, cols = struct.unpack('>IIII', features_data[:16])
    assert magic == 2051
    assert rows == 28
    assert cols == 28

    assert labels_size == features_size

    label_it = struct.iter_unpack('B', labels_data[8:])
    pixel_it = struct.iter_unpack(f'{28*28}B', features_data[16:])

    result = []
    for label, pixels in zip(label_it, pixel_it):
        label_index = label[0]
        label_bits = INDEX_TO_BITS[label_index]
        normalized_pixels = [p / 255.0 for p in pixels]
        result.append((normalized_pixels, label_bits))

    return result


def test_mnist(train_examples, test_examples):
    network = Network()
    network.add(FullyConnected(28*28, 100))
    network.add(Activation(100, sigmoid, sigmoid_derivative))
    network.add(FullyConnected(100, 50))
    network.add(Activation(50, sigmoid, sigmoid_derivative))
    network.add(FullyConnected(50, 10))
    network.add(Activation(10, sigmoid, sigmoid_derivative))

    config = TrainingConfig(
        loss=mean_squared_error,
        loss_derivative=mean_squared_error_derivative,
        epochs=1,
        learning_rate=0.5)

    train(network, config, train_examples)

    error_sum = 0
    error_count = 0

    for input_vector, expected_output in test_examples:
        output = predict(network, input_vector)
        print(f'Input={input_vector}, Output={output}')

        mse = config.loss(expected_output, output)
        error_sum += mse
        error_count += 1

    print(f'AvgError={error_sum/error_count:.10f}')


test_mnist(
    load_mnist_data(sys.argv[1], sys.argv[2]),
    load_mnist_data(sys.argv[3], sys.argv[4]))
