# CMSC471 AI Project
This is an AI algorithm to that finds out what kitchen item you have by image

## Dependencies
* Keras
* Tensorflow
* Pillow
* Numpy

## Installation
```
pip install -r requirements.txt
```

## Usage
### Testing with the given trained model
```
python predict.py 64 data/train64/ weights2.h5 test_data/blender.jpg
```
### Retraining the model
```
python train.py 64 data/train64/ data/validation64/
```
### Resizing images
```
python resize.py 64 data/train/ data/train64/
```
