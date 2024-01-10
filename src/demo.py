import mnist_loader
from network import *

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = Network([784, 60, 60, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

