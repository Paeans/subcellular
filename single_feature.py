
import sys
import numpy as np
import tensorflow as tf

from tensorflow import keras
from scipy.io import loadmat, savemat

from tensorflow.keras import layers

from sklearn.model_selection import KFold


fname = sys.argv[1]
num_folds = 5

Y_4802 = loadmat('Y_4802.mat')['Y_4802']
X_4802_feature = loadmat('feature_4802.mat')
pssmpse = X_4802_feature[fname]


kf = KFold(num_folds)

#with tf.device("cpu:0"):
for train_index, test_index in kf.split(pssmpse):
    train_x = pssmpse[train_index]
    train_y = Y_4802[train_index]

    input_shape = pssmpse.shape
    pssm = keras.Input(shape=input_shape[1:], dtype = "float32")

    x = layers.Conv2D(256, (4,3), activation='relu')(pssm)
    x = layers.Conv2D(128, 3, activation='relu',)(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu',)(x)
    #x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, activation='relu',)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(37, activation='sigmoid')(x)
    model = keras.Model(pssm, outputs)

    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    model.fit(train_x, train_y, batch_size=8, epochs=50)