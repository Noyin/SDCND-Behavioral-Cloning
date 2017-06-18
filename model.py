
# import packages

import csv
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import os
import keras
import logging

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Cropping2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from random import shuffle
### from keras.utils import plot_model


logging.basicConfig(format='%(asctime)s  %(levelname)s:%(message)s', level=logging.DEBUG)

# load dataset into samples list
logging.info('Loading data set')
np.random.seed(42)
samples = []
with open('./middel_lane_run_1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
    assert len(samples) is not 0, "Samples list is empty"


"************ split dataset into training and validation set *******"

train_samples, validation_samples = train_test_split(samples, test_size=0.3, random_state=42)
logging.info("Number of data set is {}".format(len(samples)))


# method to plot the mean squared error of the training set and validation set

def visualize_loss(history_object):
    logging.info("Plotting MSE VS EPOCH")
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['Loss', 'val_loss'], loc='upper right')
    plt.show()


# helper method to load batch of images into memory one at a time for memory management

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                # load both center camera image and a flipped copy of the center camera image

                name = batch_sample[0]
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                images.append(center_image)
                images.append(np.fliplr(center_image))

                # load both left camera image and a flipped copy of the left camera image

                name = batch_sample[1]
                left_image = cv2.imread(name)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                images.append(left_image)
                images.append(np.fliplr(left_image))

                # load both right camera image and a flipped copy of the right camera image

                name = batch_sample[2]
                right_image = cv2.imread(name)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                images.append(right_image)
                images.append(np.fliplr(right_image))

                # load both center camera angle and a negative value of the center camera image

                center_angle = float(batch_sample[3])
                angles.append(center_angle)
                center_angle_opposite = float(batch_sample[3]) * -1.0
                angles.append(center_angle_opposite)

                # load both left camera angle and a negative value of the left camera image

                left_angle = float(batch_sample[3]) + 0.2
                angles.append(left_angle)
                left_angle_opposite = left_angle * -1.0
                angles.append(left_angle_opposite)

                # load both right camera angle and a negative value of the right camera image

                right_angle = float(batch_sample[3]) - 0.2
                angles.append(right_angle)
                right_angle_opposite = right_angle * -1.0
                angles.append(right_angle_opposite)

            yield np.array(images), np.array(angles)



# method to train the model

def create_model(type_=None):

    # shape of each image in the datatset
    shape = (160, 320, 3)

    # number of training set and validation set is multiplied by 6 to account for cameras(3) and their flipped copies

    samples_per_epoch = len(train_samples) * 6
    nb_val_samples = len(validation_samples) * 6

    nb_epoch = 5

    # LENET-5 model architecture
    # the model preprocesses each image in the training set by performing normalization using the lambda layer and
    #cropping using the cropping2D layer
    # the model uses ELU to introduce non-linearity
    # the model uses a dropout layer to prevent overfitting
    logging.info("Creating LENET-5 model architecture")
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=shape))
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=shape))
    model.add(Convolution2D(6, 5, 5, activation='elu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(16, 5, 5, activation='elu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dropout(0.75))
    model.add(Dense(84))
    model.add(Dense(1))

    ### plot_model(model, to_file='model.png')

    # Callback method to save a copy of the model is there is improvement to the previously saved model

    checkpointer = ModelCheckpoint(filepath='model.h5', verbose=1, save_best_only=True)

    # Callback method to stop training if there is no improvement to validation loss

    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='auto')

    train_generator = generator(train_samples, batch_size=12)
    validation_generator = generator(validation_samples, batch_size=12)

    adam = Adam()
    model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])

    # train model and return a history object which is used to visualize the mean squared error
    # of the training set and validation set
    logging.info("Training model")
    history_object = model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch,
                                         validation_data=validation_generator, nb_val_samples=nb_val_samples,
                                         nb_epoch=nb_epoch, callbacks=[checkpointer, earlyStopping])
    return history_object

visualize_loss(create_model())
