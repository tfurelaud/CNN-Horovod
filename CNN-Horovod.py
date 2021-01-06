import pickle
import os
import sys
import glob
import tensorflow as tf
import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import horovod.tensorflow as hvd

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

hvd.init()

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_CIFAR_batch(file): 
    file_dict = unpickle(file)
    Y = file_dict[b'labels']
    X = file_dict[b'data']
    return X, Y

def load_CIFAR10(root):
    os.chdir(root)
    train_batch_list = glob.glob('data_batch*')
    test_batch = glob.glob('test_batch*')
    
    # Training set 
    x_train = ''
    y_train = ''
    for file in train_batch_list:
        x_batch, y_batch = load_CIFAR_batch(file)
        if (x_train == ''):
            x_train = x_batch
            y_train = y_batch
        else:
            x_train = np.concatenate((x_train, x_batch))
            y_train = np.concatenate((y_train, y_batch))
        
    x_train = x_train.reshape((x_train.shape[0], 3, 32, 32)).transpose(0,2,3,1)
    x_test, y_test = load_CIFAR_batch(test_batch[0])
    x_test = x_test.reshape((x_test.shape[0], 3, 32, 32)).transpose(0,2,3,1)
    
    return x_train, y_train, x_test, y_test

path_data = 'cifar-10-batches-py/'
X_train, Y_train, X_test, Y_test = load_CIFAR10(path_data) 

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


num_classes = 10
class_names =['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)

def cnn():
    model = Sequential()
    
    # Adding more layers to improve the model
    model.add(Conv2D(32, (3, 3),activation='relu', padding = 'same', input_shape=X_train.shape[1:]))
    model.add(Conv2D(32, (3, 3),activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size= (2,2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (3, 3),activation='relu', padding = 'same'))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size= (2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3),activation='relu', padding = 'same'))
    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size= (2, 2)))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(loss= 'categorical_crossentropy', optimizer = 'adam', metrics= ['accuracy'])
    return model

model = cnn()
model.summary()


batch_size = 128
epochs = 20

hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size)   

print(hist.history.keys())

score = model.evaluate(X_test, Y_test, verbose = 0)
print("Test Loss", score[0])
print("Test accuracy", score[1])

plt.figure(1)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train','validation'], loc = 'upper left')
plt.show()

plt.figure(2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("Model loss")
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(['train','validation'], loc = 'upper right')
plt.show()

