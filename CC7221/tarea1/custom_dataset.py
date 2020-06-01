import numpy as np
from tensorflow import keras
import cv2

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_files, labels, batch_size=32, dim=(256,256), n_channels=3,
                 n_classes=250, shuffle=True):
        'Initialization'
        self.data_files = data_files
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        data_files_temp = [self.data_files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(data_files_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, data_files_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(data_files_temp):
            # Store sample
            img = cv2.imread(ID, cv2.IMREAD_COLOR)
            img = cv2.resize(img, self.dim)
            X[i,] = np.array(img) / 255

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)