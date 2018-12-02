from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from PIL import Image
import numpy as np
import sys
import os

def main():
  if len(sys.argv) < 4:
    raise Exception('not enough arguments. python train.py <size> <training dir> <model> <image>')

  img_size = int(sys.argv[1])
  training_dir = sys.argv[2]
  model_name = sys.argv[3]
  image_name = sys.argv[4]
  num_of_unique_items = len(os.listdir(training_dir))

  # use a Sequential model
  model = Sequential()
  model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(img_size, img_size, 3)))
  model.add(MaxPooling2D(2, 2))

  model.add(Conv2D(32, (5, 5), activation='relu'))
  model.add(MaxPooling2D(2, 2))

  model.add(Flatten())
  model.add(Dense(1000, activation='relu'))

  model.add(Dense(num_of_unique_items,  activation='softmax'))

  model.summary()

  model.load_weights(model_name)

  # image
  img = Image.open(image_name)
  resized = img.resize((img_size, img_size), Image.ANTIALIAS)

  arr = np.array(resized).reshape((img_size, img_size, 3))
  arr = np.expand_dims(arr, axis=0)

  predict = model.predict(arr)[0]
  classes = predict.argmax(axis=-1)

  print(os.listdir(training_dir)[classes])

if __name__ == '__main__':
  main()