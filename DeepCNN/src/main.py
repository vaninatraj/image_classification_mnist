from mnist_cnn import DeepCNN
import create_data
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

print("Preparing Data...")

num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = create_data.load_data()
x_train, x_validation,  y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_validation = x_validation.reshape(x_validation.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_validation = x_validation.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_validation /= 255
x_test /= 255

y_train = to_categorical(y_train, num_classes)
y_validation = to_categorical(y_validation, num_classes)
y_test = to_categorical(y_test, num_classes)

mnist = DeepCNN(x_train, y_train, x_test, y_test, x_validation, y_validation, input_shape, num_classes)
mnist.define()
mnist.compile()
mnist.train()
mnist.test()
# mnist.predict("test.png")