import uproot
import numpy as np
import pandas as pd
import h5py

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

treename = 'Events'
filename = {}
upfile = {}
df = {}

filename['TTZ'] = '/local/mmoser/TTZ.root'
filename['TWZ'] = '/local/mmoser/TWZ_NLO_DR.root'
filename['WZ'] = '/local/mmoser/WZ.root'

#VARS = ['f_mass4l','f_massjj'] 
VARS = ['mva_Z1_eta',
        'mva_jet2_btagDeepB',
        'mva_jet0_btagDeepB',
        'mva_jet1_nonZl1_deltaR',
        'mva_m3l',
        'mva_lnonZ1_eta',
        'mva_Z1_cosThetaStar',
        'mva_jet2_pt',
        'mva_jet2_Z1_deltaR',
        'mva_jet1_Z1_deltaR',
        'mva_ht',
        'mva_met_pt',
        'mva_jet1_eta',
        'mva_jet0_Z1_deltaR',
        'mva_Z1_pt',
        'mva_nonZ1_l1_Z1_deltaPhi',
        'mva_jet0_eta',
        'mva_nJetGood',
        'mva_jet0_nonZl1_deltaR',
        'mva_nBTag',
        'mva_jet1_pt',
        'mva_maxAbsEta_of_pt30jets',
        'mva_jet2_eta',
        'mva_jet1_btagDeepB',
        'mva_bJet_Z1_deltaR',
        'mva_nonZ1_l1_Z1_deltaR',
        'mva_lnonZ1_pt',
        'mva_bJet_non_Z1l1_deltaR',
        'mva_jet0_pt',
        'mva_W_pt']




upfile['TTZ'] = uproot.open(filename['TTZ'])
upfile['TWZ'] = uproot.open(filename['TWZ'])
upfile['WZ'] = uproot.open(filename['WZ'])

df['TTZ'] = upfile['TTZ'][treename].pandas.df(branches=VARS)
df['TWZ'] = upfile['TWZ'][treename].pandas.df(branches=VARS)
df['WZ'] = upfile['WZ'][treename].pandas.df(branches=VARS)


# Plots of the Input parameters

import matplotlib.pyplot as plt


# Plot Signal vs Noise:
for i, j in enumerate(VARS):
    plt.clf()
    plt.figure(figsize=(10, 10))
    # find min and max:
    minimum1 = df['TTZ'][j].min()
    maximum1 = df['TTZ'][j].max()
    minimum2 = df['TWZ'][j].min()
    maximum2 = df['TWZ'][j].max()
    minimum3 = df['WZ'][j].min()
    maximum3 = df['WZ'][j].max()
    print(minimum2, maximum2)
    minimum = min([minimum1, minimum2, minimum3])
    maximum = max([maximum1, maximum2, maximum3])
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
    plt.savefig('plots/' + j + '.png')
    print(i+1, ' of ', len(VARS), ': ', 'plots/' + j)
    plt.close()

# Start Training

NDIM = len(VARS)

df['TWZ']['isSignal'] = np.ones(len(df['TWZ'])) * 0
df['TTZ']['isSignal'] = np.ones(len(df['TTZ'])) * 1
df['WZ']['isSignal'] =  np.ones(len(df['WZ'])) *  2



df_all = pd.concat([df['TTZ'], df['TWZ'], df['WZ']])
dataset = df_all.values
X = dataset[:,0:NDIM]
Y = dataset[:,NDIM]

from sklearn.preprocessing import label_binarize
Y = label_binarize(Y, classes=[0,1,2])

from sklearn.model_selection import train_test_split
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)


# baseline keras model
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Input, Activation, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.utils import np_utils


model = Sequential([Flatten(input_shape=(NDIM, 1)),
                    BatchNormalization(),
                    Dense(NDIM*5, activation='sigmoid'),
                    Dense(NDIM*5, activation='sigmoid'),
                    Dense(NDIM*5, activation='sigmoid'),
                    Dense(3,kernel_initializer='normal', activation='sigmoid')])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


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

test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2)

print('\nTest accuracy:', test_acc)


# save Model:
filename = 'TTZ_TWZ_WZ_keras_model_weights'


model.save("TTZ_TWZ_WZ_Keras_Model.h5")
print("Saved model to disk")


from keras.models import load_model
model = load_model("TTZ_TWZ_WZ_Keras_Model.h5")
weights = model.get_weights()

import pickle
pickle.dump(weights, open(filename + '.pkl', 'w'))
weights_loaded = 



#######################################################################################
# Plotting
#######################################################################################

# Plot ROC
Y_predict = model.predict(X_test)
from sklearn.metrics import roc_curve, auc

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_predict[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_predict.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

###########################################################################
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))


# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(3):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= 3
lw = 2
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

from itertools import cycle

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class Keras')
plt.legend(loc="lower right")
plt.savefig('plots/ROC_keras_all.png')


# confusion Matrix

Y_pred_label = np.zeros(len(Y_predict))
Y_test_label = np.zeros(len(Y_test))
for i, y in enumerate(Y_predict):
    Y_pred_label[i] = np.argmax(y)

for i, y in enumerate(Y_test):
    Y_test_label[i] = np.argmax(y)

print(Y_pred_label)
print(Y_test_label)

from sklearn.metrics import confusion_matrix

cmatkeras = confusion_matrix(Y_test_label, Y_pred_label, normalize='true')

print(cmatkeras)

# Signal vs Noise:

Y_predict_sn = np.zeros((len(Y_predict), 2))
Y_predict_sn[:,0] = Y_predict[:,0]
Y_predict_sn[:,1] = Y_predict[:,1] + Y_predict[:,2]
for i, y in enumerate(Y_predict_sn):
    Y_predict_sn[i] = Y_predict_sn[i] / np.sum(y)

# new roc curve

fpr, tpr, _ = roc_curve(Y_test[:, 0], Y_predict_sn[:, 0])
plt.clf()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class Keras')
plt.legend(loc="lower right")
plt.savefig('plots/ROC_keras_sn.png')

# Confusion Matrix with threshhold:

Y_test_label_threshhold = Y_test_label
for i, y in enumerate(Y_test_label):
    if y == 2:
        Y_test_label_threshhold[i] = 1


def confusion(threshhold): 
    #threshhold = 0.7
    Y_predict_label_threshhold = np.ones(len(Y_predict_sn))
    for i, y in enumerate(Y_predict_sn):
        if y[0] > threshhold:
            Y_predict_label_threshhold[i] = 0
    #
    cmatkeras_sn = confusion_matrix(Y_test_label, Y_predict_label_threshhold, normalize='true')
    return cmatkeras_sn

threshhold = np.linspace(0, 1, 2**7)
det = np.zeros(2**7)

for i, t in enumerate(threshhold): # takes forever, only run if oyu have a lot of time
    det[i] = np.linalg.det(confusion(t))
    print(det[i], t, i)

print(np.argmax(det))

print(confusion(threshhold[np.argmax(det)])) # maximum at ~ 0.565
