import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
import matplotlib.pyplot as plt
from preprocess_data import *
from model import unet
import segmentation_models as sm
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

lr = 1e-4
batch_size = 16
epochs = 15

img_path = "dataset/images"
mask_path = "dataset/annotations"
txt_path = "dataset/trainval.txt"

train_set, val_set, train_steps, val_steps = process_img(txt_path,img_path,mask_path)

model = unet()

model.compile(optimizer = Adam(lr = 1e-4), loss = "categorical_crossentropy")

callback= [
    tf.keras.callbacks.EarlyStopping(patience=5),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.h5'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
]

model.fit(train_set,
          epochs=epochs,
          steps_per_epoch=train_steps,
          validation_data=(val_set), 
          validation_steps=val_steps,
          callbacks=callback)
model.save("model.h5")