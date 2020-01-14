import numpy as np
import sys

split = 5000

def load_objects():
    img = np.load('data/trainX.npy')
    label = np.load('data/trainY.npy')

    train_img = img[:split]
    train_label = label[:split]
    test_img = img[split:]
    test_label = label[:split]
    print('source img shaple : ',img.shape)
    print('source label shape : ',label.shape)

    return train_img, train_label, test_img, test_label
def load_mnist_style_objects():
    img = np.load('data/testX.npy')
    label = np.load('data/dummy_testY.npy') #all_zero dummy label
    img = img.reshape(img.shape[0],img.shape[1],img.shape[2])
    print('target img shape : ',img.shape)
    print('target label (dummy) shape : ', label.shape)
    total = 100000
    train_img = img[:total]
    train_label = label[:total]
    test_img = img[:]
    test_label = label[:]

    return train_img, train_label, test_img, test_label

