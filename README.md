# Alphabet Neural Network 🧠

A lightweight, from-scratch implementation of a Deep Neural Network designed to classify handwritten letters (A-Z). \
This project uses only **NumPy** for matrix operations, demonstrating the internal logic of forward and backpropagation.


## 🚀 Key Features
* **Custom Dense Layers:** Manual implementation of weights ($W$) and biases ($b$)
* **Activation Functions:** 
	* `ReLU` for hidden layers to prevent vanishing gradients
    * `Softmax` for the output layer to provide probability distributions
* **Optimization:** Stochastic Gradient Descent (SGD) with a built-in `overfitting_penalty` (L2 Regularization)
* **Data Pipeline:** Includes a custom `OneHotEncoder` and normalization logic for image pixel data
* **Persistence:** Built-in methods to `save()` and `load()` trained model weights in `.npz` format

## 🛠️ Tech Stack
* **Python 3.x**
* **NumPy** - Linear algebra and matrix mathematics
* **Pandas** - Data loading and subset management

## 📂 Usage
1. Open `letter.png` in any image editor and write a letter (A-Z)
2. Run `testing.py`

```Python
import numpy as np
import neural_network as nn

network = nn.NeuralNetwork(data_cnt=28*28, neuron_cnt=16, classes_cnt=26, lr=0.05)
network.load("handwritten_model_weights.npz")
network.forward(X_test)
predictions = network.get_predictions()

print(f"prediction = {chr(ord('A')+np.argmax(predictions))}")
print(f"confidence = {np.max(predictions)}")
```