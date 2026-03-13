from PIL import Image
import numpy as np
import neural_network as nn

def img_to_array(imgname):
    img = Image.open(imgname).convert('L')  # load and convert to grayscale
    img = img.resize((28, 28))
    return np.array(img).flatten()

def make_label(label):
    return [ord(label)-ord('A')]


# normalize
img_array = img_to_array("letter.png").astype('float32') / 255.0
# data
X_test = img_array.reshape(1, 784)
# y = make_label('O')
# test
network = nn.NeuralNetwork(data_cnt=28*28, neuron_cnt=16, classes_cnt=26, lr=0.05)
network.load("handwritten_model_weights.npz")
network.forward(X_test)
predictions = network.get_predictions()
# output
# print(f"label = {chr(ord('A')+y[0])}")
print(f"prediction = {chr(ord('A')+np.argmax(predictions))}")
print(f"confidence = {np.max(predictions)}")
# print(predictions)
