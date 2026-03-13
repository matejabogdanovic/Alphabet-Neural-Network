import numpy as np
import numpy.random
import pandas as pd

class NeuronLayer:
    def __init__(self, input, output):
        self.weights = np.random.randn(input, output) * 0.001
        self.biases = np.zeros((1, output))

    def forward(self, x):
        self.inputs = x
        self.output = np.dot(x, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class ReLUActivation:
    def forward(self, x):
        self.inputs = x
        self.output = np.maximum(0, x)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class SoftmaxActivation:
    def forward(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


class OneHotEncoder:
    def encode(self, x):
        m = np.zeros((x.size, x.max()+1))
        m[np.arange(x.size), x] = 1
        return m


class NeuralNetwork:

    def __init__(self, data_cnt, neuron_cnt, classes_cnt, lr):
        self.layer1 = NeuronLayer(input=data_cnt, output=neuron_cnt)
        self.layer2 = NeuronLayer(input=neuron_cnt, output=classes_cnt)
        self.softmax = SoftmaxActivation()
        self.relu = ReLUActivation()
        self.LR = lr

    def forward(self, X):
        self.layer1.forward(X)
        self.relu.forward(self.layer1.output)
        self.layer2.forward(self.relu.output)
        self.softmax.forward(self.layer2.output)

    def backward(self, y):
        self.softmax.backward(self.softmax.output, y)
        self.layer2.backward(self.softmax.dinputs)
        self.relu.backward(self.layer2.dinputs)
        self.layer1.backward(self.relu.dinputs)

    def update(self, penalty):
        self.layer1.weights += -self.LR * (self.layer1.dweights + penalty * self.layer1.weights)
        self.layer1.biases += -self.LR * (self.layer1.dbiases)
        self.layer2.weights += -self.LR * (self.layer2.dweights + penalty * self.layer2.weights)
        self.layer2.biases += -self.LR * (self.layer2.dbiases)

    def get_predictions(self):
        return self.softmax.output

    def train(self, iteration_cnt, filename ,samples, normalization_divisor, overfitting_penalty):
        # reading data
        df = pd.read_csv(filename)
        subset_df = df.groupby(df.columns[0]).head(samples)

        X_train = np.array(subset_df.iloc[:, 1:]).astype('float32') / normalization_divisor
        y = np.array(subset_df.iloc[:, 0])
        y_enc = OneHotEncoder().encode(y)

        accuracy = 0
        for i in range(iteration_cnt):
            self.forward(X_train)
            # print acc and loss
            if (i % (iteration_cnt / 10)) == 0:
                predictions = np.argmax(self.softmax.output, axis=1)
                accuracy = np.mean(predictions == y)

                loss = -np.mean(np.log(self.softmax.output[np.arange(len(y)), y] + 1e-7))
                print(f"acc = {accuracy} loss = {loss}")

            self.backward(y_enc)
            self.update(overfitting_penalty)
        return accuracy

    def save(self, filename):
        np.savez(filename,
                 w1=self.layer1.weights,
                 b1=self.layer1.biases,
                 w2=self.layer2.weights,
                 b2=self.layer2.biases)

    def load(self, filename):
        model_data = np.load(filename)

        self.layer1.weights = model_data['w1']
        self.layer1.biases = model_data['b1']
        self.layer2.weights = model_data['w2']
        self.layer2.biases = model_data['b2']


if __name__ == "__main__":
    network = NeuralNetwork(data_cnt=28*28, neuron_cnt=16, classes_cnt=26, lr=0.09)
    accuracy = network.train(iteration_cnt=2000,
                             filename="A_Z Handwritten Data.csv",
                             samples=1000,
                             normalization_divisor=255.0,
                             overfitting_penalty=0.001)

    if accuracy >= 0.8:
        network.save("handwritten_model_weights.npz")
        print("Model saved.")
