import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
import matplotlib.pyplot as plt
from preprocess_data import *
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

img_path = "dataset/images"
mask_path = "dataset/annotations"
txt_path = "dataset/trainval.txt"

x_train, x_val, y_train, y_val = process_img(txt_path,img_path,mask_path, batch=False)
load_model = tf.keras.models.load_model("model.h5")

for idx, (img, mask) in enumerate(zip(x_train[:10], y_train[:10])):
    read_mask = np.expand_dims(mask,axis=-1)
    read_mask = read_mask*(255/3)
    read_mask = read_mask.astype(np.int32)
    read_mask = np.concatenate([read_mask,read_mask,read_mask],axis=2)

    pred = load_model.predict(np.expand_dims(img,axis=0))[0]
    pred = np.argmax(pred, axis=1)
    pred = np.expand_dims(pred,axis=-1)
    pred = pred*(255/3)
    pred = pred.astype(np.int32)
    pred = np.concatenate([pred, pred, pred],axis=2)

    read_img = (read_img*255.0)
    read_img = img.astype(np.int32)

    line = np.ones((128,10,3))*255

    final_img = np.concatenate([read_img,line,read_mask,line,pred],axis=1)
    cv2.imwrite(f"results/{idx}",final_img)

