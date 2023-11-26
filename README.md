Basic neural network code to understand the limits of Python performance for an HPC workload.

To run the XOR model that shows the pieces are generally working properly:

```shell
python3 ./xor_model.py
```

To run the MNIST model, get the [data here](https://www.kaggle.com/code/hojjatk/read-mnist-dataset/input), adjust the function arguments like output path in the `mnist_model.py` file, and then run:

```shell
python3 ./mnist_model.py train-images.idx3-ubyte train-labels.idx1-ubyte t10k-images.idx3-ubyte t10k-labels.idx1-ubyte
```

I was able to reach an accuracy of 89% after a short time training. I figure better hyperparameters or an Adam optimizer would help it learn better. L2 regularization had a huge impact on how well it converged.
