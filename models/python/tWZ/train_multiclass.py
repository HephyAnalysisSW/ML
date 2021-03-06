import argparse
parser = argparse.ArgumentParser()
parser.add_argument("config_path", help="enter the name of the config file")

config = parser.parse_args().config_path
print(config)

# import config file
import importlib
import sys

try:
    module = importlib.import_module(config)
except:
    print('File was not able to be imported.')
    print('Use command to train:')
    print('python train_multiclass.py tWZ')
    print('Check if list of variables is configured correctly')
    sys.exit(1)

##########################################################################################

err_list = []

try:
    variables = module.variables
except:
    err_list.append('variables')

try:
   treename = module.treename    
except:
    err_list.append('treename')

try:
    filename = module.filename
except:
    err_list.append('filename')

try:
    model_path = module.model_path
except:
    err_list.append('model_path')

try:
    save_path = module.save_path
except:
    err_list.append('save_path')

try:
    NL = module.NL
except:
    err_list.append('NL')
    
try:
    make_plots = module.make_plots
except:
    err_list.append('make_plots')

try:
    make_roc = module.make_roc
except:
    err_list.append('make_roc')

try:
    make_conf = module.make_conf
except:
    err_list.append('make_conf')

try:
    batch_size = module.batch_size
except:
    err_list.append('batch_size')

try:
    make_impo = module.make_impo
except:
    err_list.append('make_impo')

try:
    make_outp = module.make_outp
except:
    err_list.append('make_outp')

try:
    make_acc_loss = module.make_acc_loss
except:
    err_list.append('make_acc_loss')

if err_list !=[]:
    print('One or more required variables in the config file missing.')
    print('missing variable list:')
    print(err_list)
    print()
    print('variables     -> variables of the root tree, will be input variables for the training')
    print('treename      -> name of the root tree')
    print('filename      -> a dict with class names as key and path to root file as values')
    print('model_path    -> the path where to save the model')
    print('save_path     -> the path where to save the plots')
    print('NL            -> Network layout, list of number of nodes in hidden layer')
    print('make_plots    -> bool if histogram plots of all input variables should be made')
    print('make_roc      -> bool if roc-curve  plot should be made')
    print('make_conf     -> bool if confusion matrix should be made')
    print('batch_size    -> batch size in training and model evaluation')
    print('make_impo     -> make feature importance plots')
    print('make_outp     -> make output plots')
    print('make_acc_loss -> make ccuracy and loss plot of the training history')
    sys.exit(1)


import uproot
import numpy as np
import pandas as pd
import h5py

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

upfile = {}
df = {}

# read data into dataframe
key_list = list(filename.keys())

for key in key_list: # root file to pandas dataframe
    upfile[key] = uproot.open(filename[key]) 
    df[key] = upfile[key][treename].pandas.df(branches=variables)

# preprocessing
NDIM = len(variables) # number of variables
N_classes = len(filename) # number of classes

class_digit = range(N_classes)

for key, digit in zip(key_list, class_digit): # add isSignal variable, class_digit
    df[key]['isSignal'] = np.ones(len(df[key])) * digit


# concatenate the dataframes
df_all = pd.concat([df[key_list[0]], df[key_list[1]]])

if len(filename) > 1:
    for i in range(2, len(filename)):
        df_all = pd.concat([df_all, df[key_list[i]]])

# check for NaN in the dataframe, .root file might be slightly broken
for key in key_list:
    for var in variables:
        if df[key][var].isnull().values.any() == True:
            print(key,':', var, ' has some entries as nan:')
            print(df[key][var].isnull().sum(), ' are nan')
            print('nan Events will be removed')

df_all = df_all.dropna() # removes all Events with nan

# split dataset into Input and output data
dataset = df_all.values
X = dataset[:,0:NDIM]
Y = dataset[:,NDIM]


# make histogram plots of the input variables
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
        for key in filename:
            a, b, c = plt.hist(df[key][j], bins=30, range=(minimum, maximum), label=key,
                density=True, histtype=u'step', linewidth=2)
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

# split data into train and test, test_size = 0.2 is quite standard for this
from sklearn.model_selection import train_test_split
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7, shuffle = True)

# define model (neural network)
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

# define callback for early stopping
import tensorflow as tf
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)

# train the model
history = model.fit(X_train_val, 
                    Y_train_val, 
                    epochs=1000, 
                    batch_size=batch_size,
                    #verbose=0, # switch to 1 for more verbosity, 'silences' the output
                    callbacks=[callback],
                    #validation_split=0.1
                    validation_data=(X_test,Y_test) # use either validation_split or validation_data
                   )
print('trainig finished')

# saving
model.save(model_path + config + '_keras_model.h5')


# make accuracy vs epoch and loss vs epoch plots
if make_acc_loss:
    print('Star accuracy and loss plot')
    import matplotlib.pyplot as plt
    plt.clf()
    plt.figure(figsize=(15, 6))
    # score
    ax = plt.subplot(1, 2, 1)
    ax.plot(history.history['loss'], label='loss')
    ax.plot(history.history['val_loss'], label='val_loss')
    ax.legend(loc="upper right")
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title('loss')
    # accuraccy
    ax = plt.subplot(1, 2, 2)
    ax.plot(history.history['accuracy'], label='acc')
    ax.plot(history.history['val_accuracy'], label='val_acc')
    ax.legend(loc="lower right")
    ax.set_xlabel('epoch')
    ax.set_ylabel('acc')
    ax.set_title('accuracy')
    #
    plt.savefig(save_path + 'acc_score.png')


# make a plot of the roc curves for each variable
if make_roc:
    print('Start roc-plot')
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
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'indigo', 'darkgreen', 'crimson',
        'sienna', 'darkmagenta', 'darkslatergrey', 'maroon', 'olive', 'purple'])
    for i, color in zip(range(len(filename)), colors):
        ra = int(roc_auc[i] * 100)
        ra = ra / 100
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=key_list[i] + ' (area = ' + str(ra) +')')
    #
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class Keras')
    plt.legend(loc="lower right")
    plt.savefig(save_path + 'ROC_curve.png')


# print the confusion matrix
if make_conf:
    Y_predict = model.predict(X_test)
    print('Start confusion matrix')
    Y_pred_label = np.zeros(len(Y_predict))
    Y_test_label = np.zeros(len(Y_test))
    for i, y in enumerate(Y_predict):
        Y_pred_label[i] = np.argmax(y)
    #
    for i, y in enumerate(Y_test):
        Y_test_label[i] = np.argmax(y)
    #
    from sklearn.metrics import confusion_matrix
    #
    cmat = confusion_matrix(Y_test_label, Y_pred_label, normalize='true')
    #
    print(np.array(cmat))


# calculate the permutation variable importance
if make_impo:
    print('Start importance')
    #
    shuff_list = []
    scores = model.evaluate(X_train_val, Y_train_val, verbose=0, batch_size = 1024*8)
    #
    for i, var in enumerate(variables):
        print(i+1, 'of', len(variables))
        X_train_shuff = X_train_val.copy()
        X_train_shuff = np.array(X_train_shuff)
        np.random.shuffle(X_train_shuff[:,i])
        scores_shuff = model.evaluate(X_train_shuff, Y_train_val, verbose=0, batch_size = 1024*8)
        shuff_list.append(scores_shuff)
    #
    dif_ac = [] #accuracy
    dif_sc = [] #score
    #
    for sh in shuff_list:
        dif_sc.append(sh[0] - scores[0])
        dif_ac.append(scores[1] - sh[1])
    #
    var_ac = variables.copy()
    var_sc = variables.copy()
    #
    dif_ac, var_ac = (list(t) for t in zip(*sorted(zip(dif_ac, var_ac), reverse = True)))
    dif_sc, var_sc = (list(t) for t in zip(*sorted(zip(dif_sc, var_sc), reverse = True)))
    #
    dif_ac = np.array(dif_ac)
    dif_sc = np.array(dif_sc)
    dif_ac /= np.max(dif_ac)
    dif_sc /= np.max(dif_sc)
    #
    for i, a, s in zip(range(len(dif_ac)), dif_ac, dif_sc):
        dif_ac[i] = int(a * 10000) / 10000
        dif_sc[i] = int(s * 10000) / 10000
    #
    import matplotlib.pyplot as plt
    #
    def autolabel(rects, bar_label):
        for idx,rect in enumerate(bar_plot):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    bar_label[idx],
                    ha='center', va='bottom', rotation=90)
    # accuracy
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots()
    bar_plot = plt.bar(list(range(len(var_ac))), dif_ac)
    autolabel(bar_plot, var_ac)
    plt.ylim(0,2)
    plt.title('Permutation importance of variables with accuracy as metric')
    plt.savefig("/local/mmoser/plots/importance_accuracy.png")
    # score
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots()
    bar_plot = plt.bar(list(range(len(var_sc))), dif_sc)
    autolabel(bar_plot, var_sc)
    plt.ylim(0,2)
    plt.title('Permutation importance of variables with score as metric')
    plt.savefig("/local/mmoser/plots/importance_score.png")


# make output plots
if make_outp:
    print('Start output plot')
    import matplotlib.pyplot as plt
    for i, key in enumerate(key_list):
        print(i+1, 'of', len(key_list), 'plots')
        plt.clf()
        counter = 1
        for key2 in key_list:
            print(counter, 'of', len(key_list), 'histograms')
            counter += 1
            df_new = df[key2].copy()
            df_new = df_new.dropna()
            #
            dataset = df_new.values
            X = dataset[:,0:NDIM]
            Y = dataset[:,NDIM]
            #
            from sklearn.preprocessing import label_binarize
            classes = range(len(filename))
            Y = label_binarize(Y, classes=classes)
            #
            Y_predict = model.predict(X)
            #
            a, b, c = plt.hist(Y_predict[:, i], range=(0, 1), density=True, histtype=u'step',
                linewidth=2, bins=30, label=key2)
        #
        plt.xlabel('probability')
        plt.ylabel('N')
        plt.yscale('log')
        plt.legend()
        plt.title(key + ' output')
        plt.savefig('/local/mmoser/plots/output_plot_' + key +'.png')



