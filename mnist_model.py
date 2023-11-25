import concurrent.futures
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
    for i, (label, pixels) in enumerate(zip(label_it, pixel_it)):
        label_index = label[0]
        label_bits = INDEX_TO_BITS[label_index]
        normalized_pixels = [p / 255.0 for p in pixels]
        result.append((normalized_pixels, label_bits))

    return result


def argmax(vector):
    it = iter(vector)
    max_index = 0
    max_value = next(it)
    for i, value in enumerate(it, 1):
        if value > max_value:
            max_index = i
            max_value = value

    return max_index


def create_network():
    network = Network(PARAMETER_SIZE_BYTES)
    network.add(FullyConnected(28*28, 100))
    network.add(Activation(100, sigmoid, sigmoid_derivative))
    network.add(FullyConnected(100, 50))
    network.add(Activation(50, sigmoid, sigmoid_derivative))
    network.add(FullyConnected(50, 10))
    network.add(Activation(10, sigmoid, sigmoid_derivative))
    return network


def train_mnist(train_examples, output_path, *, resume_path=None):
    network = create_network()

    parameters = network.allocate_parameters()
    network.connect(parameters)

    print(f'Model parameters are {len(parameters)} bytes')

    if resume_path:
        with open(resume_path, 'rb') as f:
            f.readinto(parameters)
    else:
        network.initialize()

    # A bigger batch size is better when the number of model parameters is
    # larger, since the whole model will be serialized for each shard that's
    # processed in parallel. Larger batches will amortize that serialization.
    config = TrainingConfig(
        loss=mean_squared_error,
        loss_derivative=mean_squared_error_derivative,
        epochs=1,
        batch_size=1024,
        parallelism=8,
        learning_rate=0.1)

    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=config.parallelism)

    train(network, config, executor, train_examples)

    print(f'Outputting model checkpoint to {output_path!r}')
    with open(output_path, 'wb') as f:
        f.write(parameters)


def eval_mnist(test_examples, resume_path):
    network = create_network()

    parameters = network.allocate_parameters()
    network.connect(parameters)

    with open(resume_path, 'rb') as f:
        f.readinto(parameters)

    loss = mean_squared_error

    error_sum = 0
    error_count = 0
    correct_count = 0

    test_examples = random.sample(test_examples, 100)

    for input_vector, expected_output in test_examples:
        output = predict(network, input_vector)
        mse = loss([expected_output], output)
        error_sum += sum(mse)
        error_count += len(mse)

        expected_argmax = argmax(expected_output)
        found_argmax = argmax(output[0])
        correct = found_argmax == expected_argmax
        if correct:
            correct_count += 1

        print(
            f'Example={error_count}, '
            f'Found={found_argmax}, '
            f'Expected={expected_argmax}, '
            f'Correct={correct}, '
            f'Error={mse}')

        print('Label:  ', ', '.join('%.1f' % o for o in expected_output))
        print('Output: ', ', '.join('%.1f' % o for o in output[0]))
        print()


    print(
        f'AvgError={error_sum/error_count:.10f}, '
        f'CorrectPercentage={100 * correct_count / error_count:.2f}%')


if __name__ == '__main__':
    train_mnist(
        load_mnist_data(sys.argv[1], sys.argv[2]),
        'mnist.bin5')
        # resume_path='mnist.bin3')

    eval_mnist(
        load_mnist_data(sys.argv[3], sys.argv[4]),
        'mnist.bin5')
