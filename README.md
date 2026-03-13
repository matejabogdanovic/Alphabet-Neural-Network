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
1. Draw a letter (A-Z) and save it as 'letter.png' in root directory.
2. Run `testing.py`.