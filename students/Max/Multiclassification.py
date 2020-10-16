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
    plt.title(j, fontsize=20)
    plt.ylabel('N', fontsize=20)
    plt.xlabel('Energy [GeV]', fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig('plots/' + j + '.png')
    print(i+1, ' of ', len(VARS), ': ', 'plots/' + j)


# bis hier 16.10.


# baseline keras model
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Input, Activation, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.utils import np_utils

NDIM = len(VARS)
inputs = Input(shape=(NDIM,), name = 'input')  
outputs = Dense(1, name = 'output', kernel_initializer='normal', activation='sigmoid')(inputs)

# creae the model
# choose the default model or the one with 1 hidden layer
model = Model(inputs=inputs, outputs=outputs)
# or:
model = Sequential([Flatten(input_shape=(NDIM, 1)),
                    Dense(NDIM, activation='sigmoid'),
                    Dense(1,kernel_initializer='normal', activation='sigmoid')])

# or: 
#with Batch Normalisation

model = Sequential([Flatten(input_shape=(NDIM, 1)),
                    BatchNormalization(),
                    Dense(NDIM, activation='sigmoid'),
                    Dense(NDIM, activation='sigmoid'),
                    Dense(NDIM, activation='sigmoid'),
                    Dense(1,kernel_initializer='normal', activation='sigmoid')])


# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# print the model summary
model.summary()


df_all = df['data']
#df_all = pd.concat([df['VV'],df['bkg']])
dataset = df_all.values
X = dataset[:,0:NDIM]
Y = dataset[:,NDIM]

from sklearn.model_selection import train_test_split
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

# preprocessing: standard scalar
from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler().fit(X_train_val)
#X_train_val = scaler.transform(X_train_val)
#X_test = scaler.transform(X_test)



import tensorflow as tf

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

history = model.fit(X_train_val, 
                    Y_train_val, 
                    epochs=1000, 
                    batch_size=1024*4,
                    #verbose=0, # switch to 1 for more verbosity, 'silences' the output
                    callbacks=[callback],
                    #validation_split=0.25
                    validation_data=(X_test,Y_test)
                   )




#%matplotlib inline
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

# Plot ROC
Y_predict = model.predict(X_test)
from sklearn.metrics import roc_curve, auc
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
print(df_all.iloc[:5])