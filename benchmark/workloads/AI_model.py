# Simple feedforward neural network for AI benchmarking
import numpy as np


def relu(x):
    return np.maximum (0, x)


def forward(weights, biases, x):
    layer1 = relu (np.dot (weights [0], x) + biases [0])
    layer2 = np.dot (weights [1], layer1) + biases [1]
    return layer2


def main():
    # Dummy data: 2-layer MLP with 784->128->10 neurons
    weights = [np.random.randn (128, 784), np.random.randn (10, 128)]
    biases = [np.random.randn (128), np.random.randn (10)]
    x = np.random.randn (784)

    for _ in range (1000):
        output = forward (weights, biases, x)

    np.savetxt ("output.txt", output)


if __name__ == "__main__":
    main ()
