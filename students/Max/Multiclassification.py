import uproot
import numpy as np
import pandas as pd
import h5py
import pickle

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


########################################################################################
## sklearn Teil
########################################################################################
#
#
#NDIM = len(VARS)
#
#df['TWZ']['isSignal'] = np.ones(len(df['TWZ'])) * 0
#df['TTZ']['isSignal'] = np.ones(len(df['TTZ'])) * 1
#df['WZ']['isSignal'] = np.ones(len(df['WZ'])) * 2
#df_all = pd.concat([df['TWZ'], df['TTZ'], df['WZ']])
#dataset = df_all.values
#X = dataset[:,0:NDIM]
#Y = dataset[:,NDIM]
#
#
#
## splitting the 3 Data sets into Train and Test Parts
#from sklearn.model_selection import train_test_split
#
#X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
#
## preprocessing: standard scalar
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler().fit(X_train_val)
#X_train_val = scaler.transform(X_train_val)
#X_test = scaler.transform(X_test)
#
#from sklearn.multiclass import OneVsOneClassifier
#from sklearn.neural_network import MLPClassifier
#
#clf = OneVsOneClassifier(MLPClassifier(solver='adam', alpha=1e-5,
#    hidden_layer_sizes=(5,5), random_state=7, early_stopping=True,
#    max_iter=100000, activation='logistic'))
#clf.fit(X_train_val, Y_train_val)
#
#print(clf.predict(X_test[:20]))
#print(Y_test[:20])
#testlen = len(Y_test)
#falsch = 0
#richtig = 0
#predicted = clf.predict(X_test)
#for i in range(0, testlen):
#    if Y_test[i] == predicted[i]:
#        richtig += 1
#    else:
#        falsch += 1
#
#
#skit_accuracy = richtig/testlen
#print('Richtig Klassifiziert:', skit_accuracy)
#print(richtig, falsch, testlen)
#print(clf.predict(X_test))
#
#
########################################################################################
## Save trained Model:
#
#pickle.dump(clf, open("TTZ_TWZ_WZ_sklearn_Model.sav", 'wb'))
#pickle.dump(scaler, open("TTZ_TWZ_WZ_sklearn_skaler.sav", 'wb'))
#
##load
#sklearn_model_loaded = pickle.load(open("TTZ_TWZ_WZ_sklearn_Model.sav", 'rb'))
#scaler_loaded = pickle.load(open("TTZ_TWZ_WZ_sklearn_skaler.sav", 'rb'))
##result = sklearn_model_loaded.score(X_test, Y_test)
##print(result)
##X_test = scaler_loaded.transform(X_test)
#pred = sklearn_model_loaded.predict(X_test[:3])
#print(pred)
#
########################################################################################
## Plot
#
#Y_predict = clf.decision_function(X_test)
#
#from sklearn.metrics import roc_curve, auc
#
#from sklearn.preprocessing import label_binarize
#Y_test_bin = label_binarize(Y_test, classes=[0,1,2])
#
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#for i in range(3):
#    fpr[i], tpr[i], _ = roc_curve(Y_test_bin[:, i], Y_predict[:, i])
#    roc_auc[i] = auc(fpr[i], tpr[i])
#
## Compute micro-average ROC curve and ROC area
#fpr["micro"], tpr["micro"], _ = roc_curve(Y_test_bin.ravel(), Y_predict.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
#
#
## First aggregate all false positive rates
#all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
#
#
## Then interpolate all ROC curves at this points
#mean_tpr = np.zeros_like(all_fpr)
#for i in range(3):
#    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
#
## Finally average it and compute AUC
#mean_tpr /= 3
#lw = 2
#fpr["macro"] = all_fpr
#tpr["macro"] = mean_tpr
#roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
## Plot all ROC curves
#plt.figure()
#plt.plot(fpr["micro"], tpr["micro"],
#         label='micro-average ROC curve (area = {0:0.2f})'
#               ''.format(roc_auc["micro"]),
#         color='deeppink', linestyle=':', linewidth=4)
#
#plt.plot(fpr["macro"], tpr["macro"],
#         label='macro-average ROC curve (area = {0:0.2f})'
#               ''.format(roc_auc["macro"]),
#         color='navy', linestyle=':', linewidth=4)
#
#from itertools import cycle
#
#colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
#for i, color in zip(range(3), colors):
#    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#             label='ROC curve of class {0} (area = {1:0.2f})'
#             ''.format(i, roc_auc[i]))
#
#plt.plot([0, 1], [0, 1], 'k--', lw=lw)
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Some extension of Receiver operating characteristic to multi-class sklearn')
#plt.legend(loc="lower right")
#plt.savefig('plots/ROC_sklearn_all.png')

NDIM = len(VARS)

#######################################################################################
# Compare with Keras One versus All:
#######################################################################################

df['TWZ']['isSignal'] = np.ones(len(df['TWZ'])) * 0
df['TTZ']['isSignal'] = np.ones(len(df['TTZ'])) * 1
df['WZ']['isSignal'] = np.ones(len(df['WZ'])) * 2



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
                    batch_size=1024*6,
                    #verbose=0, # switch to 1 for more verbosity, 'silences' the output
                    callbacks=[callback],
                    #validation_split=0.25
                    validation_data=(X_test,Y_test)
                   )

test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2)

print('\nTest accuracy:', test_acc)
#print('Skit accuracy:', skit_accuracy)

#######################################################################################
# Saving and loading trained Model:

# Keras
filename = "TTZ_TWZ_WZ_Keras_Model"
model.save(filename + '.h5')
with open( filename + '.pkl', 'wb') as f:
    pickle.dump( model.get_weights(), f, protocol=2)
print("Saved model to disk")


# keras
from keras.models import load_model

keras_model_loaded = load_model("TTZ_TWZ_WZ_Keras_Model.h5")
pred = keras_model_loaded.predict(X_test[:3])
print(pred)

#######################################################################################
# Plotting
#######################################################################################

'''
# plot loss vs epoch
plt.figure(figsize=(15,10))
ax = plt.subplot(2, 2, 1)
ax.plot(history.history['loss'], label='loss')
ax.plot(history.history['val_loss'], label='val_loss')
ax.legend(loc="upper right")
ax.set_xlabel('epoch')
ax.set_ylabel('loss')

# plot accuracy vs epoch
ax = plt.subplot(2, 2, 2)
ax.plot(history.history['accuracy'], label='acc')
ax.plot(history.history['val_accuracy'], label='val_acc')
ax.legend(loc="upper left")
ax.set_xlabel('epoch')
ax.set_ylabel('acc')
'''
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
'''
#eine Einzelne ROC - Kurve
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('plots/ROC_keras.png')
'''
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




'''
#######################################################
fpr, tpr, thresholds = roc_curve(Y_test, Y_predict)
roc_auc = auc(fpr, tpr)
ax = plt.subplot(2, 2, 3)
ax.plot(fpr, tpr, lw=2, color='cyan', label='auc = %.3f' % (roc_auc))
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='random chance')
ax.set_xlim([0, 1.0])
ax.set_ylim([0, 1.0])
ax.set_xlabel('false positive rate')
ax.set_ylabel('true positive rate')
ax.set_title('receiver operating curve')
ax.legend(loc="lower right")
plt.show()


df_all['dense'] = model.predict(X) # add prediction to array
print(df_all.iloc[:5])'''

