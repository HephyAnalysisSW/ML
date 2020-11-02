import argparse
parser = argparse.ArgumentParser()
parser.add_argument("config_path", help="enter the name of the config file")

args = parser.parse_args().config_path
config = args.config_path

##########################################################################################
import uproot
import numpy as np
import pandas as pd
import h5py

# import config file
import importlib
module = importlib.import_module(config)

variables = module.variables
treename = module.treename
filename = module.filename
model_path = module.model_path
save_path = module.save_path
NL = module.NL
make_plots = module.make_plots
make_roc = module.make_roc
make_conf = module.make_conf

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

upfile = {}
df = {}

# read data into dataframe

for key in filename:
    upfile[key] = uproot.open(filename[key])
    df[key] = upfile[key][treename].pandas.df(branches=variables)

# preprocessing
NDIM = len(variables)
N_classes = len(filename)

class_digit = range(N_classes)

for key, digit in zip(filename, class_digit):
    df[key]['isSignal'] = np.ones(len(df[key])) * digit

# concatenate the dataframes
key_list = list(filename.keys())
df_all = pd.concat([df[key_list[0]], df[key_list[1]]])

if len(filename) > 1:
    for i in range(2, len(filename)):
        df_all = pd.concat([df_all, df[key_list[i]]])

df_all = df_all.dropna() # removes all Events with nan

dataset = df_all.values
X = dataset[:,0:NDIM]
Y = dataset[:,NDIM]

# make plots

if make_plots:
    import matplotlib.pyplot as plt
    for i, j in enumerate(variables):
        plt.clf()
        plt.figure(figsize=(10, 10))
        # find min and max:
        maximuml = []
        minimuml = []
        for key in filename:
            maximuml.append(df[key][j].max())
            minimuml.append(df[key][j].min())
        minimum = min(minimuml)
        maximum = max(maximuml)
        # make histograms
        a, b, c = plt.hist(df['TTZ'][j], bins=30, range=(minimum, maximum), label='TTZ',
          density=True, histtype=u'step', linewidth=2)
        a, b, c = plt.hist(df['TWZ'][j], bins=30, range=(minimum, maximum), label='TWZ',
          density=True, histtype=u'step', linewidth=2, alpha=0.8)
        a, b, c = plt.hist(df['WZ'][j], bins=30, range=(minimum, maximum), label='WZ',
          density=True, histtype=u'step', linewidth=2, alpha=0.5)
        xlab = j
        xlab = xlab.replace('mva_', '')
        xlab = xlab.replace('_', ' ')
        plt.title(xlab, fontsize=20)
        plt.ylabel('N', fontsize=20)
        plt.xlabel(xlab, fontsize=20)
        plt.legend(fontsize=15)
        plt.savefig(save_path + j + '.png')
        print(i+1, ' of ', len(variables), ': ', 'plots/' + j)
        plt.close()




# binarize class labels 
from sklearn.preprocessing import label_binarize
classes = range(len(filename))
Y = label_binarize(Y, classes=classes)

from sklearn.model_selection import train_test_split
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

# define model
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Input, Activation, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.utils import np_utils

model = Sequential()
model.add(BatchNormalization(input_shape=(NDIM, )))

for dim in NL:
    model.add(Dense(dim, activation='sigmoid'))

model.add(Dense(len(filename), kernel_initializer='normal', activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# train the model
import tensorflow as tf
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)

history = model.fit(X_train_val, 
                    Y_train_val, 
                    epochs=1000, 
                    batch_size=1024,
                    #verbose=0, # switch to 1 for more verbosity, 'silences' the output
                    callbacks=[callback],
                    #validation_split=0.25
                    validation_data=(X_test,Y_test)
                   )

# saving

model.save(model_path + 'keras_model.h5')


# trainig finished, now plotting roc and calculating confusion matrix

if make_roc:
    import matplotlib.pyplot as plt
    Y_predict = model.predict(X_test)
    from sklearn.metrics import roc_curve, auc
    #
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(filename)):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    #
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_predict.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    #
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(filename))]))
    #
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(filename)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    #
    # Finally average it and compute AUC
    mean_tpr /= len(filename)
    lw = 2
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    #
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    #
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    #
    from itertools import cycle
    #
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(len(filename)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=key_list[i] + ' area =' + str(roc_auc[i]))
    #
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class Keras')
    plt.legend(loc="lower right")
    plt.savefig(save_path + 'ROC_curve.png')


if make_conf:
    Y_pred_label = np.zeros(len(Y_predict))
    Y_test_label = np.zeros(len(Y_test))
    for i, y in enumerate(Y_predict):
        Y_pred_label[i] = np.argmax(y)
    #
    for i, y in enumerate(Y_test):
        Y_test_label[i] = np.argmax(y)
    #
    print(Y_pred_label)
    print(Y_test_label)
    #
    from sklearn.metrics import confusion_matrix
    #
    cmat = confusion_matrix(Y_test_label, Y_pred_label, normalize='true')
    #
    print(np.array(cmat))



