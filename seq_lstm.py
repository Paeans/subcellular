
import numpy as np
import tensorflow as tf

from tensorflow import keras
from scipy.io import loadmat, savemat

from tensorflow.keras import layers

from sklearn.model_selection import KFold
from sklearn.metrics import label_ranking_average_precision_score as avgprec
from sklearn.metrics import coverage_error, label_ranking_loss

num_folds = 10


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
    
Y_4802 = loadmat('Y_4802.mat')['Y_4802']
sequence = loadmat('dataset_4802.mat')['Sequence']
amino_code = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6,
             'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 
             'Q':13, 'R':14, 'S':15, 'T':16, 'U':17, 'V':18,
             'W':19, 'X':20, 'Y':21 }

extend_one_encode = []
seq_num_encode = []
for s in sequence:
    p_seq = (list(s[0][0]) * 1000)[0:10000]
    seq_num = np.array([amino_code[x] for x in p_seq])
    seq_num_encode.append(seq_num)
    tmp = np.zeros((seq_num.size, 22))
    tmp[np.arange(seq_num.size), seq_num] = 1
    extend_one_encode.append(tmp)
    
X_4802_num = np.array(seq_num_encode)

kf = KFold(num_folds)

ap_list = []
rl_list = []
ce_list = []

#with tf.device("cpu:0"):
for train_index, test_index in kf.split(Y_4802):
    train_x = X_4802_num[train_index]
    train_y = Y_4802[train_index]
    
    inputs = keras.Input(shape=(None,), dtype = "int32")
    x = layers.Embedding(22, 22)(inputs)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences = True))(x)
    #x = layers.Bidirectional(layers.LSTM(256, return_sequences = True))(x)
    x = layers.Bidirectional(layers.LSTM(128))(x)
    x = layers.Reshape((16,16,1))(x)

    x = layers.Conv2D(256, (4,3), activation='relu')(x)
    x = layers.Conv2D(128, 3, activation='relu',)(x)
    # x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu',)(x)
    # x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, activation='relu',)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(37, activation='sigmoid')(x)

    # x = layers.Conv2D(128, 3, activation='relu',)(x)
    # x = layers.Conv2D(128, 3, activation='relu',)(x)
    # x = layers.Conv2D(64, 3, activation='relu',)(x)
    # x = layers.Conv2D(32, 3, activation='relu',)(x)
    # x = layers.Flatten()(x)
    # outputs = layers.Dense(37, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    
    model.compile("adam", "binary_crossentropy", metrics=["binary_accuracy"])
    model.fit(train_x, train_y, batch_size=8, epochs=2,)
    
    test_x = X_4802_num[test_index]
    test_y = Y_4802[test_index]
    
    pred_y = model.predict(test_x)
    ap_score = avgprec(test_y, pred_y)
    ap_list.append(ap_score)
    rl_list.append(label_ranking_loss(test_y, pred_y))
    ce_list.append(coverage_error(test_y, pred_y) - 1)
    
with open('seq_4802_res.txt', 'w') as result_file:    
    result_file.write('the ap score is: ' + str(ap_list) + '\n')
    result_file.write('average is: {}'.format(sum(ap_list)/len(ap_list)) + '\n')

    result_file.write('the rl score is: ' + str(rl_list) + '\n')
    result_file.write('average is: {}'.format(sum(rl_list)/len(rl_list)) + '\n')

    result_file.write('the ce score is: ' + str(ce_list) + '\n')
    result_file.write('average is: {}'.format(sum(ce_list)/len(ce_list)) + '\n')