
import numpy as np
import tensorflow as tf

from tensorflow import keras
from scipy.io import loadmat, savemat

from tensorflow.keras import layers

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)
    
Y_4802 = loadmat('Y_4802.mat')['Y_4802']

sequence = loadmat('dataset_4802.mat')['Sequence']
amino_code = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6,
             'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 
             'Q':13, 'R':14, 'S':15, 'T':16, 'U':17, 'V':18,
             'W':19, 'X':20, 'Y':21 }

extend_one_encode = []
for s in sequence:
    p_seq = (list(s[0][0]) * 1000)[0:10000]
    seq_num = np.array([amino_code[x] for x in p_seq])
    tmp = np.zeros((seq_num.size, 22))
    tmp[np.arange(seq_num.size), seq_num] = 1
    extend_one_encode.append(tmp)
    
X_4802_padding = np.array(extend_one_encode)

inputs = keras.Input(shape=(None, 22,), dtype = "float32")
x = layers.Bidirectional(layers.LSTM(128, return_sequences = True))(inputs)
#x = layers.Bidirectional(layers.LSTM(128, return_sequences = True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
outputs = layers.Dense(37, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)
model.summary()

model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
model.fit(X_4802_padding, Y_4802, batch_size=32, epochs=2)#, validation_data=(x_val, y_val))