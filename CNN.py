# @ Neeraj Tripathi

# Data is organised as per the 'dataset' directory structure
# Data Source :  https://www.superdatascience.com/machine-learning/
# Importing components of the CNN

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Building the CNN

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(3,3), strides=2, padding='same', input_shape=(128, 128, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), strides=1, activation = 'relu'))
model.add(AveragePooling2D(pool_size = (2, 2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), strides=1, activation = 'relu'))
model.add(AveragePooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(units=128, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(units=64, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(units=32, activation='relu', kernel_initializer='glorot_uniform'))

model.add(Dense(units=1, activation='sigmoid', kernel_initializer='glorot_uniform'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')
import datetime
print()
print('Start time : ' + str(datetime.datetime.now()))
print()

model.fit_generator(training_set,
                    steps_per_epoch = 250,
                    epochs = 25,
                    validation_data = test_set,
                    validation_steps = 80)
print()
print('Finish time : ' + str(datetime.datetime.now()))
print()
