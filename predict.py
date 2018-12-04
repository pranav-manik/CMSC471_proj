from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from PIL import Image
import numpy as np
import sys
import os

def main():
  if len(sys.argv) < 4:
    raise Exception('not enough arguments. python predict.py <size> <training dir> <model> <image>')

  img_size = int(sys.argv[1])
  training_dir = sys.argv[2]
  model_name = sys.argv[3]
  image_name = sys.argv[4]
  num_of_unique_items = len(os.listdir(training_dir))

  """We are using the same model from train.py"""
  model = Sequential()
  model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(img_size, img_size, 3)))
  model.add(MaxPooling2D(2, 2))

  model.add(Conv2D(32, (5, 5), activation='relu'))
  model.add(MaxPooling2D(2, 2))

  model.add(Flatten())
  model.add(Dense(1000, activation='relu'))

  model.add(Dense(num_of_unique_items,  activation='softmax'))

  """Outputs a nice little summary"""
  model.summary()

  """Load in our saved weights"""
  model.load_weights(model_name)

  """We want our image that we want to classify
  to be the same size we used in our training"""
  img = Image.open(image_name)
  resized = img.resize((img_size, img_size), Image.ANTIALIAS)

  """Reshape our array"""
  arr = np.array(resized).reshape((img_size, img_size, 3))
  arr = np.expand_dims(arr, axis=0)

  """Predict! (The cool part)"""
  predict = model.predict(arr)[0]
  """Get the index with the highest probability"""
  classes = predict.argmax(axis=-1)

  """Output the prediction.
  The labels are not saved, but the order is in the
  alphabetical order of the categories. So the directory
  is a good way to get the name. The directory has to be sorted
  since depending on the filesystem, this may or may not be
  in alphabetical order"""
  print(sorted(os.listdir(training_dir))[classes])
  
  """Close the image handler"""
  img.close()

if __name__ == '__main__':
  main()
