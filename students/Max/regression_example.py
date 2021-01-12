import uproot
import numpy as np
import pandas as pd
#import h5py

#########################################################################################
# variable definitions
variables = ['mva_Z1_eta',
            'mva_jet2_btagDeepB',
            'mva_Z1_pt',
            'mva_jet0_btagDeepB',
            'mva_jet1_nonZl1_deltaR',
            'mva_m3l',
            'mva_lnonZ1_eta',
            'mva_Z1_cosThetaStar',
            'mva_jet2_pt',
            'mva_jet1_Z1_deltaR',
            'mva_ht',
            'mva_met_pt',
            'mva_maxAbsEta_of_pt30jets',
            'mva_jet0_Z1_deltaR',
            'mva_lnonZ1_pt',
            'mva_jet2_Z1_deltaR',
            'mva_jet0_eta',
            'mva_nJetGood',
            'mva_jet0_nonZl1_deltaR',
            'mva_nBTag',
            'mva_jet1_pt',
            'mva_nonZ1_l1_Z1_deltaPhi',
            'mva_jet1_eta',
            'mva_jet2_eta',
            'mva_jet1_btagDeepB',
            'mva_bJet_Z1_deltaR',
            'mva_nonZ1_l1_Z1_deltaR',
            'mva_bJet_non_Z1l1_deltaR',
            'mva_jet0_pt',
            'mva_W_pt',
            'mva_l1_mvaTOP',
            'mva_l2_mvaTOP',
            'mva_l3_mvaTOP']

treename = 'Events'
filename = {}

batch_size = 1024*4

# key is used as name on the plot
filename['TWZ'] = '/local/mmoser/root_files_2/TWZ_NLO_DR.root'
filename['TTZ'] = '/local/mmoser/root_files_2/TTZ.root'
filename['WZ'] = '/local/mmoser/root_files_2/WZ.root'
filename['NON'] = '/local/mmoser/root_files_2/nonprompt_3l.root'

# model savepath:
model_path = '/local/mmoser/'

# for the plots
save_path = '/local/mmoser/plots/'

NDIM = len(variables)

# Network layout:
NL = [NDIM*5, NDIM*5, NDIM*5]
#########################################################################################
# define some function to be regressed:

def to_regress(values): # some random function (the result is not random)
    '''Values must be a list of a list'''
    y_list = []
    for val in values:
        y = 0
        for i, v in enumerate(val):
            if i > 0:
                y += v / i**2
            else:
                y += 1
        y_list.append(y)
    return y_list

#########################################################################################
# read data and preprocessing

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

for key, digit in zip(key_list, class_digit): # add fun variable and values, this is pretty slow
    print(key, 'of', key_list)
    df[key]['fun'] = to_regress(df[key].values)


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

# split data into train and test, test_size = 0.2 is quite standard for this
from sklearn.model_selection import train_test_split
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7, shuffle = True)


#########################################################################################
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

model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_percentage_error'])
model.summary()

# define callback for early stopping
import tensorflow as tf
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1) # patience can be higher if a more accurate result is preferred
                                                                        # I would recommmend at least 3, otherwise it might cancel too early

# train the model
history = model.fit(X_train_val, 
                    Y_train_val, 
                    epochs=100, 
                    batch_size=batch_size,
                    #verbose=0, # switch to 1 for more verbosity, 'silences' the output
                    callbacks=[callback],
                    #validation_split=0.1
                    validation_data=(X_test,Y_test) # use either validation_split or validation_data
                   )
print('trainig finished')

# saving
model.save(model_path + '_regression_model.h5')

#########################################################################################
# Apply the model
print(model.predict(X_test[9:10]), Y_test[9:10])
