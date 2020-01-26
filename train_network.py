import tensorflow as tf
print("Keras version:", keras.__version__)
print("TensorFlow version:", tf.__version__)
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import preprocess_input
from keras.preprocessing import image
import numpy as np
from glob import glob
from keras.preprocessing.image import ImageDataGenerator
#Load the pre-trained model
#add preprocessing layer to the front of DenseNet
densenet = DenseNet121(
    weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',
    include_top=False,
    input_shape=(224,224,3)
)

# No need training for existing weights
for layer in densenet.layers:
  layer.trainable = False
  
train_dir = './Datasets/Train/train'
validation_dir = './clean-dataset/validation'

  # This is needed for getting number of classes
folders = glob('Datasets/Train/*')
  # our layers - you can add more if you want
x = Flatten()(densenet.output) ##https://stackoverflow.com/questions/43237124/what-is-the-role-of-flatten-in-keras
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=densenet.input, outputs=prediction)
# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
