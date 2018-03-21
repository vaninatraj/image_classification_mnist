import numpy as np
import pandas as pd
import os
import urllib.request
import time
import sys

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def download_file(url, filename):
    urllib.request.urlretrieve(url, filename, reporthook)

def download_if_not_exists(url, filename):
    if not os.path.exists(filename):
        download_file(url, filename)
        return True
    return False
        
def generate_training_data():
    dataframe = pd.read_csv(os.path.join("..", "data", "mnist_train.csv"))
    labels = dataframe.iloc[:, 0]
    data = dataframe.iloc[:, 1:]

    labels = np.array(labels)
    data = np.array(data)
    data = data.reshape(data.shape[0], 28, 28)
    return (data, labels)

def generate_testing_data():
    dataframe = pd.read_csv(os.path.join("..", "data", "mnist_test.csv"))
    labels = dataframe.iloc[:, 0]
    data = dataframe.iloc[:, 1:]

    labels = np.array(labels)
    data = np.array(data)
    data = data.reshape(data.shape[0], 28, 28)
    return (data, labels)

def load_data():
    download_if_not_exists("http://www.pjreddie.com/media/files/mnist_train.csv", os.path.join("..", "data", "mnist_train.csv"))
    download_if_not_exists("http://www.pjreddie.com/media/files/mnist_test.csv", os.path.join("..", "data", "mnist_test.csv"))
    (x_train, y_train) = generate_training_data()
    (x_test, y_test) = generate_testing_data()
    return (x_train, y_train), (x_test, y_test)