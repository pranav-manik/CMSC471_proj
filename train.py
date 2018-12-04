from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
import sys
import os

BATCH_SIZE = 10
EPOCHS = 30

def main():
  if len(sys.argv) < 4:
    raise Exception('not enough arguments. python train.py <size> <training dir> <validation dir>')

  img_size = int(sys.argv[1])
  training_dir = sys.argv[2]
  validation_dir = sys.argv[3]
  num_of_unique_items = len(os.listdir(training_dir))

  """Use a Sequential model
  Sequential model is a linear stack of layers
  https://keras.io/models/sequential/"""
  model = Sequential()

  """Add a Convolution 2D layer with 16 filters, a 5x5 kernel size
  and use ReLU as our activation function
  https://keras.io/layers/convolutional/#conv2d"""
  model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(img_size, img_size, 3)))

  """Use max pooling to downsample
  https://keras.io/layers/pooling/#maxpooling2d"""
  model.add(MaxPooling2D(2, 2))

  """Repeat"""
  model.add(Conv2D(32, (5, 5), activation='relu'))
  model.add(MaxPooling2D(2, 2))

  """Flatten the input
  https://keras.io/layers/core/#flatten"""
  model.add(Flatten())

  """Add a Dense layer with 1000 units
  https://keras.io/layers/core/#dense. Also using
  ReLU as our activation function"""
  model.add(Dense(1000, activation='relu'))

  """Another dense layer with the number of items
  we would like to classify as our units. Also using
  softmax as our activation function"""
  model.add(Dense(num_of_unique_items,  activation='softmax'))

  """Create our model using binary crossentropy as our loss function,
  optmize our loss with rmsprop, and use accuracy as our main metric"""
  model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

  """We have a shortage of data, so we would like to expand on that
  by adding more variables to our existing data"""
  training_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    fill_mode='nearest',
    horizontal_flip=True
  )

  """We want our validation data to be normal, but different scaling
  is not a bad idea"""
  validation_datagen = ImageDataGenerator(
    rescale=1./255
  )

  """Pass the images into our data generator"""
  training_gen = training_datagen.flow_from_directory(
    training_dir,
    target_size=(img_size, img_size),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
  )

  validation_gen = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_size, img_size),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
  )

  """Train!"""
  model.fit_generator(
    training_gen,
    verbose=1,
    epochs=EPOCHS,
    steps_per_epoch=3000 // BATCH_SIZE,
    validation_steps=5,
    validation_data=validation_gen
  )

  """Save our weights for later use"""
  model.save_weights('weights2.h5')

if __name__ == '__main__':
  main()
