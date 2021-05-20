import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

def batch(x, y, batch=16):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle()
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
        read_img /= 255.0
        read_img = read_img.astype(np.float32)
        
        read_mask = cv2.resize(cv2.imread(x,cv2.IMREAD_GRAYSCALE),(128,128))
        read_mask -= 1
        read_mask = read_mask.astype(np.int32)

        return read_img, read_mask

    img, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    masks = tf.one_hot(mask, 3, dtype=tf.int32)
    img.set_shape([128, 128, 3])
    masks.set_shape([128, 128, 3])

    return img, masks


def process_img(txt_path, img_path, mask_path, split_pct=0.15, batch_size=32):
    x_train = []
    y_train = []

    df = pd.read_csv(txt_path, sep=" ", header=None)
    filenames = df[0].values

    imgs = [os.path.join(img_path, f"{name}.jpg") for name in filenames]
    masks = [os.path.join(mask_path, f"{name}.png") for name in filenames]

    for img, mask in zip(imgs, masks):
        try: 
            read_img = cv2.resize(cv2.imread(img,cv2.IMREAD_COLOR),(128,128))
            read_mask = cv2.resize(cv2.imread(mask,cv2.IMREAD_GRAYSCALE),(128,128))
            read_img /= 255.0
            read_mask -= 1
            read_img = read_img.astype(np.float32)
            read_mask = read_mask.astype(np.int32)
            x_train.append(read_img)
            y_train.append(read_mask)
        except Exception as e:
            pass

    split_idx = int(len(x_train)*split_pct)

    x_val = x_train[-split_idx:]
    x_test = x_train[-split_idx*2:-split_idx]

    y_val = y_train[-split_idx:]
    y_test = y_train[-split_idx*2:-split_idx]

    x_train = x_train[:split_idx*2]
    y_train = y_train[:split_idx*2]

    x_train = np.array(x_train)
    x_val = np.array(x_val)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    train_set = batch(x_train,y_train)
    val_set = batch(x_val, y_val)
    test_set = batch(x_test, y_test)

    return train_set, val_set, test_set

def vis_img(img):
    plt.imshow(img)
    plt.show()