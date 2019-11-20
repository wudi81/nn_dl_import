import mnist_loader

import network

# a=[1,2,3]
# b=[4,5,6]
# c=zip(a,b)
# c = list(c)
# print(c)
# print(len(c))
# print(c)


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])

net.SGD(training_data, 30, 10, 5.0,test_data=test_data)