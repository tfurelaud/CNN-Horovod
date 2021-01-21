import pickle
import os
import sys
import glob
import tensorflow as tf
import keras
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

# Import horovod library
import horovod.keras as hvd

# Initalizing horovod
hvd.init()

# Pin a GPU to each workers
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


# These 3 first functions are there to load CIFAR data from local
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


# Load CIFAR-10 data
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


# CNN architecture
def cnn():
    model = Sequential()
    
    # First basic layer
    model.add(Conv2D(32, (3, 3),activation='relu', padding = 'same', input_shape=X_train.shape[1:]))
    model.add(Conv2D(32, (3, 3),activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size= (2,2)))
    model.add(Dropout(0.2))
    
    # Second layer
    model.add(Conv2D(64, (3, 3),activation='relu', padding = 'same'))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size= (2, 2)))
    model.add(Dropout(0.3))

    # Third layer
    model.add(Conv2D(128, (3, 3),activation='relu', padding = 'same'))
    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size= (2, 2)))
    model.add(Dropout(0.4))
    
    # Flattening the model
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,activation='softmax'))

    # Adjusting learning rate based on number of GPUs
    opt = keras.optimizers.Adam(0.001 * hvd.size())
    # Add Horovod Distributed Optimizer
    opt = hvd.DistributedOptimizer(opt)
    # Compiling the model with the distributed optimizer
    model.compile(loss= 'categorical_crossentropy', optimizer = opt, metrics= ['accuracy'])
    return model

model = cnn()
model.summary()

# Broadcast initial variable states from rank 0 to all other processes.
# This is necessary to ensure consistent initialization of all workers when
# training is started with random weights or restored from a checkpoint.
callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

# Save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))


batch_size = 128
# 20 epochs to see stagntation in results (Could be optimized)
epochs = 20

# Fiting the model
hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size, verbose=1 if hvd.rank()==0 else 0)   

print(hist.history.keys())

score = model.evaluate(X_test, Y_test, verbose = 0)
print("Test Loss", score[0])
print("Test accuracy", score[1])
