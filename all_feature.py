
import sys
import numpy as np
import tensorflow as tf

from tensorflow import keras
from scipy.io import loadmat, savemat

from tensorflow.keras import layers

from sklearn.model_selection import KFold
from sklearn.metrics import label_ranking_average_precision_score as avgprec
from sklearn.metrics import coverage_error, label_ranking_loss

num_folds = 10

Y_4802 = loadmat('Y_4802.mat')['Y_4802']
X_4802_feature = loadmat('feature_4802.mat')

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

kf = KFold(num_folds)

ap_list = []
rl_list = []
ce_list = []

#with tf.device("cpu:0"):
for train_index, test_index in kf.split(Y_4802):
    train_x = []
    train_y = Y_4802[train_index]

    model_list = []    
    test_x = []
    
    for fname in ['ppab', 'ppdwt', 'pppse', 'pssmab', 'pssmdwt', 'pssmpse']:
        fdata = X_4802_feature[fname][train_index]
        train_x.append(fdata)
        test_x.append(X_4802_feature[fname][test_index])
        
        input_shape = fdata.shape
        ix = keras.Input(shape=input_shape[1:], dtype = "float32")

        x = layers.Conv2D(256, (4,3), activation='relu')(ix)
        x = layers.Conv2D(128, 3, activation='relu',)(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, activation='relu',)(x)
        #x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(32, 3, activation='relu',)(x)
        outputs = layers.Flatten()(x)
        #outputs = layers.Dense(37, activation='sigmoid')(x)
        model = keras.Model(ix, outputs)
        model_list.append(model)

    x = layers.concatenate([m.output for m in model_list])
    x = layers.Dense(37, activation='sigmoid')(x)
    model = keras.Model(inputs=[m.input for m in model_list], outputs = x)
    model.compile("adam", "binary_crossentropy", metrics=["binary_accuracy"])
    model.fit(train_x, train_y, batch_size=8, epochs=50)
    
    test_y = Y_4802[test_index]    
    pred_y = model.predict(test_x)
    
    ap_list.append(avgprec(test_y, pred_y))
    rl_list.append(label_ranking_loss(test_y, pred_y))
    ce_list.append(coverage_error(test_y, pred_y) - 1)
    
with open('all_4802_res.txt', 'w') as result_file:    
    result_file.write('the ap score is: ' + str(ap_list) + '\n')
    result_file.write('average is: {}'.format(sum(ap_list)/len(ap_list)) + '\n')

    result_file.write('the rl score is: ' + str(rl_list) + '\n')
    result_file.write('average is: {}'.format(sum(rl_list)/len(rl_list)) + '\n')

    result_file.write('the ce score is: ' + str(ce_list) + '\n')
    result_file.write('average is: {}'.format(sum(ce_list)/len(ce_list)) + '\n')