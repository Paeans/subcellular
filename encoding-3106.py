
import numpy as np
import tensorflow as tf

from tensorflow import keras
from scipy.io import loadmat, savemat

from tensorflow.keras import layers, regularizers

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
    
from sklearn.model_selection import KFold
from sklearn.metrics import label_ranking_average_precision_score as avgprec
from sklearn.metrics import coverage_error, label_ranking_loss

num_folds = 5
    
Y_4802 = loadmat('Y_3106.mat')['Y_3106']

sequence = loadmat('dataset_3106.mat')['sequence_3106']
amino_code = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6,
             'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 
             'Q':13, 'R':14, 'S':15, 'T':16, 'U':17, 'V':18,
             'W':19, 'X':20, 'Y':21 }

# extend_one_encode = []
# for s in sequence:
#     p_seq = (list(s[0][0]) * 1000)[0:10000]
#     seq_num = np.array([amino_code[x] for x in p_seq])
#     tmp = np.zeros((seq_num.size, 22))
#     tmp[np.arange(seq_num.size), seq_num] = 1
#     extend_one_encode.append(tmp)
def preprocess(p_seq, l):
    seq_num = np.array([amino_code[x] for x in p_seq])
    tmp = np.zeros((l, 22))
    s = (100 - seq_num.size)//2
    e = s + seq_num.size
    tmp[np.arange(s, e), seq_num] = 1
    return tmp

def p_split(p_seq, l):
    result = []
    s = 0
    e = s + l
    while (s + l//2) < len(p_seq):
        t = p_seq[s:e]
        if len(t) < l:
            t = p_seq[-l:]
        s = s + l//2
        e = s + l
        result.append(t)
    return result

uniform_one_encode = []
uniform_label = []
for sq, lb in zip(sequence, Y_4802):
    seq = sq[0][0]
    if len(seq) > 100:
        result = p_split(seq, 100)
    else:
        result = [seq]
    for r in result:
        uniform_one_encode.append(preprocess(r, 100))
        uniform_label.append(lb)

    
X_4802_uni = np.array(uniform_one_encode)
Y_4802_uni = np.array(uniform_label)

kf = KFold(num_folds, shuffle = True)

ap_list = []
rl_list = []
ce_list = []

count = 0
#with tf.device("cpu:0"):
for train_index, test_index in kf.split(Y_4802_uni):
    train_x = X_4802_uni[train_index]
    train_y = Y_4802_uni[train_index]
    
    test_x = X_4802_uni[test_index]
    test_y = Y_4802_uni[test_index]
    
    inputs = keras.Input(shape=(100, 22,), dtype = "float32")
    x = layers.Bidirectional(layers.LSTM(256, return_sequences = True))(inputs)
    #x = layers.Bidirectional(layers.LSTM(128, return_sequences = True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences = True))(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(14, activation='sigmoid',
                           kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
    model = keras.Model(inputs, outputs)
    model.summary()

    model.compile("adam", "binary_crossentropy", metrics=["binary_accuracy", "binary_crossentropy"])
#     model.fit(train_x, train_y, batch_size=16, epochs=2)#, validation_data=(x_val, y_val))    
    for i in range(10):
        model.fit(train_x, train_y, batch_size=32, epochs=3, verbose=2)
        pred_y = model.predict(test_x)
        
        savemat('result_3106_' + str(count) + '_' + str(i) + '.mat', {'pred_y':pred_y, 'test_y':test_y})

        ap_list.append(avgprec(test_y, pred_y))
        rl_list.append(label_ranking_loss(test_y, pred_y))
        ce_list.append(coverage_error(test_y, pred_y) - 1)
        
        print('ap_list: {}'.format(ap_list))
        print('rl_list: {}'.format(rl_list))
        print('ce_list: {}'.format(ce_list))
    count += 1
    
ap_values = np.array(ap_list).reshape((5,20))
rl_values = np.array(rl_list).reshape((5,20))
ce_values = np.array(ce_list).reshape((5,20))
    
with open('new_encoding_3106_uni.txt', 'w') as result_file:    
    result_file.write('the ap score is: \n')
    result_file.write(str(ap_values) + '\n')
    result_file.write('max is: {}'.format(np.amax(ap_values, axis = 1)) + '\n')    
    result_file.write('k-fold is: {}'.format(np.average(ap_values, axis = 0)) + '\n')
    result_file.write('k-fold max is: {}'.format(np.amax(np.average(ap_values, axis = 0))) + '\n')

    result_file.write('the rl score is: \n')
    result_file.write(str(rl_values) + '\n')
    result_file.write('min is: {}'.format(np.amin(rl_values, axis = 1)) + '\n')    
    result_file.write('k-fold is: {}'.format(np.average(rl_values, axis = 0)) + '\n')
    result_file.write('k-fold min is: {}'.format(np.amin(np.average(rl_values, axis = 0))) + '\n')

    result_file.write('the ce score is: \n')
    result_file.write(str(ce_values) + '\n')
    result_file.write('min is: {}'.format(np.amin(ce_values, axis = 1)) + '\n')    
    result_file.write('k-fold is: {}'.format(np.average(ce_values, axis = 0)) + '\n')
    result_file.write('k-fold min is: {}'.format(np.amin(np.average(ce_values, axis = 0))) + '\n')