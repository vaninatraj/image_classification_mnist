from keras.layers import MaxPooling2D, Conv2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model, save_model
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import os
import h5py
import create_data

class DeepCNN:
    def __init__(self):
        print("Preparing Data")
        self.num_classes = 10
        (x_train, y_train), (x_test, y_test) = create_data.load_data()
        x_train, x_validation,  y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2)
        
        self.x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        self.x_validation = x_validation.reshape(x_validation.shape[0], 28, 28, 1)
        self.x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

        self.x_train = self.x_train.astype('float32')
        self.x_validation = self.x_validation.astype('float32')
        self.x_test = self.x_test.astype('float32')

        self.x_train /= 255
        self.x_validation /= 255
        self.x_test /= 255

        self.y_train = to_categorical(y_train, self.num_classes)
        self.y_validation = to_categorical(y_validation, self.num_classes)
        self.y_test = to_categorical(y_test, self.num_classes)

    def define(self):
        model = Sequential()
        model.add(Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'))
        model.add(Conv2D(64, (3,3), activation='relu'))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3,3), activation='relu'))
        model.add(Conv2D(128, (3,3), activation='relu'))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        self.model = model
        return self

    def compile(self, epochs=10):
        self.optimzer = Adam(lr=0.01)
        self.loss = 'categorical_crossentropy'
        self.epochs = epochs
        self.model.compile(optimizer=self.optimzer, loss=self.loss, metrics=['accuracy'])

    def train(self, batch_size=32):
        self.batch_size = batch_size
        self.model.fit(self.x_train, self.y_train, epochs=self.epochs, validation_data=(self.x_validation, self.y_validation), batch_size=self.batch_size, verbose=2)
        save_model(self.model, os.path.join("..", "models", "trained_model.h5"))
    
    def test(self):
        model = load_model(os.path.join("..", "models", "trained_model.h5"))
        acc, loss = model.evaluate(self.x_test, self.y_test)
        print("Test Accuarcy : {0}".format(acc))
        print("Test Loss : {0}".format(loss))

    def predict(self, path):
        self.img = load_img(path, target_size=(28, 28))
        self.img = img_to_array(self.img)
        self.img = self.img.astype('float32')
        self.img /= 255
        self.img = np.expand_dims(self.img, axis=0)

        model = load_model(os.path.join("..", "models", "trained_model.h5"))
        predicted = model.predict(self.img)
        print("Predicted Class : {0}".format(predicted))

mnist = DeepCNN()
mnist.define()
mnist.compile()
mnist.train()
mnist.test()
mnist.predict("test.png")