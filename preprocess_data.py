import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def batch(x, y, batch=16):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(2)
    return dataset

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()

        read_img = cv2.resize(cv2.imread(x,cv2.IMREAD_COLOR),(128,128))
        read_img = read_img.astype(np.float32)
        read_img /= 255.0
        
        read_mask = cv2.resize(cv2.imread(y,cv2.IMREAD_GRAYSCALE),(128,128))
        read_mask = read_mask.astype(np.int32)
        read_mask -= 1


        return read_img, read_mask

    img, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    mask = tf.one_hot(mask, 3, dtype=tf.int32)
    img.set_shape([128, 128, 3])
    mask.set_shape([128, 128, 3])

    return img, mask


def process_img(txt_path, img_path, mask_path, split_pct=0.15, batch_size=32, batch=True):

    df = pd.read_csv(txt_path, sep=" ", header=None)
    filenames = df[0].values

    imgs = [os.path.join(img_path, f"{name}.jpg") for name in filenames]
    masks = [os.path.join(mask_path, f"{name}.png") for name in filenames]

    x_train, x_val = train_test_split(imgs, test_size=split_pct, random_state=42)
    y_train, y_val = train_test_split(masks, test_size=split_pct, random_state=42)

    if batch:
        train_set = batch(x_train,y_train)
        val_set = batch(x_val, y_val)

        return train_set, val_set, int(len(x_train)//batch_size), int(len(x_val)//batch_size)
    else:
        train_x = []
        train_y = []
        val_x = []
        val_y = []
        for img,mask in zip(x_train, y_train):
            read_img = cv2.resize(cv2.imread(img,cv2.IMREAD_COLOR),(128,128))
            read_img = read_img.astype(np.float32)
            read_img /= 255.0
            
            read_mask = cv2.resize(cv2.imread(mask,cv2.IMREAD_GRAYSCALE),(128,128))
            read_mask = read_mask.astype(np.int32)
            read_mask -= 1

            train_x.append(read_img)
            train_y.append(read_mask)

        for img,mask in zip(x_val, y_val):
            read_img = cv2.resize(cv2.imread(img,cv2.IMREAD_COLOR),(128,128))
            read_img = read_img.astype(np.float32)
            read_img /= 255.0
            
            read_mask = cv2.resize(cv2.imread(mask,cv2.IMREAD_GRAYSCALE),(128,128))
            read_mask = read_mask.astype(np.int32)
            read_mask -= 1

            val_x.append(read_img)
            val_y.append(read_mask)

        return train_x, val_x, train_y, val_y


def vis_img(img):
    plt.imshow(img)
    plt.show()