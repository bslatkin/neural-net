import concurrent.futures
from neural_net import *


def test_xor():
    network = Network()
    network.add(FullyConnected(2, 3))
    network.add(Activation(3, sigmoid, sigmoid_derivative))
    network.add(FullyConnected(3, 1))
    network.add(Activation(1, sigmoid, sigmoid_derivative))
    network.initialize()

    config = TrainingConfig(
        loss=mean_squared_error,
        loss_derivative=mean_squared_error_derivative,
        epochs=10_000,
        batch_size=4,
        parallelism=1,
        learning_rate=0.1,
        l2_regularization=0.0001)

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=config.parallelism)

    labeled_examples = [
        ((0, 0), (0,)),
        ((0, 1), (1,)),
        ((1, 0), (1,)),
        ((1, 1), (0,)),
    ]

    train(network, config, executor, labeled_examples)

    test_examples = labeled_examples

    error_sum = 0
    error_count = 0

    for input_vector, expected_output in test_examples:
        output = predict(network, input_vector)
        print(f'Input={input_vector}, Output={output}')

        expected_matrix = np.array([expected_output])
        mse = config.loss(expected_matrix, output)
        error_sum += sum(mse)
        error_count += len(mse)

    print(f'AvgError={error_sum/error_count:.10f}')


if __name__ == '__main__':
    test_xor()
