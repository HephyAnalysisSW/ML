variables = ['mva_Z1_eta',
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

output_specification = ['TWZ', 'TTZ', 'WZ']
NDIM = len(variables)


from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Input, Activation, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.utils import np_utils

NDIM = len(variables)

model = Sequential([BatchNormalization(input_shape=(NDIM, )),
                    Dense(NDIM*5, activation='sigmoid'),
                    Dense(NDIM*5, activation='sigmoid'),
                    Dense(NDIM*5, activation='sigmoid'),
                    Dense(3,kernel_initializer='normal', activation='sigmoid')
                    ])
import pickle, os
import numpy as np
local_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))

filename = "TTZ_TWZ_WZ_Keras_Model.pkl"
# this must work in python 2 and 3 :-)
full_filename = os.path.join( local_dir, filename)
with open( full_filename, 'rb') as f:
    model.set_weights(pickle.load(f))
    print ("Loaded weights from %s"%full_filename)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

import os
import numpy as np
from keras.models import load_model
local_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
model = load_model(os.path.join( local_dir, "TTZ_TWZ_WZ_Keras_Model.h5") )

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

if __name__ == "__main__":
    inputs = [[2.2721688747406006, -10.0, -10.0, -1.0, 263.69891357421875, -0.9794921875,
            -0.03838849067687988,
            0.0, -1.0, -1.0, 0.0, 72.53070831298828, -10.0, -1.0, 7.966670989990234,
            2.1394686698913574, -10.0, 0.0, -1.0, 0.0, 0.0, -1.0, -10.0, -10.0,
            -1.0, 3.892380475997925, 93.152099609375, -1.0, 0.0, 39.12520980834961]]
    pred = model.predict(np.array(inputs))
    print(pred)