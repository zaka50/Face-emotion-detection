import os
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization


train_dir = '/content/emotionclass/train'
validation_dir = '/content/emotionclass/validation'

train_generator = ImageDataGenerator(rescale= 1./255,
                                     horizontal_flip = True,
                                     fill_mode = 'nearest',
                                     shear_range = 0.3,
                                     width_shift_range=0.4,
                                     height_shift_range=0.4,
                                     rotation_range = 30)

train_gen = train_generator.flow_from_directory(train_dir,
                                                target_size = (48, 48),
                                                batch_size = 32,
                                                class_mode = 'categorical',
                                                color_mode = 'grayscale',
                                                shuffle = True)

validation_generator = ImageDataGenerator(rescale = 1./255)

validation_gen = validation_generator.flow_from_directory(validation_dir,
                                                          target_size = (48, 48),
                                                          batch_size = 32,
                                                          class_mode = 'categorical',
                                                          color_mode = 'grayscale',
                                                          shuffle = True)


emotion = Sequential()
emotion.add(Conv2D(32,(3,3), padding = 'same', input_shape = (48, 48, 1),  activation= 'relu'))
emotion.add(Conv2D(32,(3,3), padding = 'same', activation= 'relu'))
emotion.add(BatchNormalization())
emotion.add(MaxPooling2D(pool_size=(2,2), strides= (2,2)))
emotion.add(Dropout(0.2))

emotion.add(Conv2D(64,(3,3), padding = 'same', activation= 'relu'))
emotion.add(Conv2D(64,(3,3), padding = 'same', activation= 'relu'))
emotion.add(BatchNormalization())
emotion.add(MaxPooling2D(pool_size=(2,2), strides= (2,2)))
emotion.add(Dropout(0.2))

emotion.add(Conv2D(128,(3,3), padding = 'same', activation= 'relu'))
emotion.add(Conv2D(128,(3,3), activation= 'relu'))
emotion.add(BatchNormalization())
emotion.add(MaxPooling2D(pool_size=(2,2), strides= (2,2)))
emotion.add(Dropout(0.2)

emotion.add(Conv2D(256,(3,3), padding = 'same', activation= 'relu'))
emotion.add(Conv2D(256,(3,3), padding = 'same', activation= 'relu'))
emotion.add(BatchNormalization())
emotion.add(MaxPooling2D(pool_size=(2,2), strides= (2,2)))
emotion.add(Dropout(0.2))

emotion.add(Flatten())
emotion.add(Dense(1024, activation = 'relu'))
emotion.add(Dropout(0.4))

emotion.add(Dense(2048, activation = 'relu'))
emotion.add(Dropout(0.4))

emotion.add(Dense(4096, activation = 'relu'))
emotion.add(BatchNormalization())


emotion.add(Dense(5, activation = 'softmax'))

emotion.compile(loss = 'categorical_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])

history = emotion.fit(train_gen,
                      steps_per_epoch = 24282 // 32,
                      epochs = 150,
                      validation_data = validation_gen,
                      validation_steps = 5937 // 32)

emotion.save('Emotion_saved.h5')