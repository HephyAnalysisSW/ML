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

from keras.models import load_model
model = load_model("TTZ_TWZ_WZ_keras_model.h5")

#if __name__ == "__main__":
#    print(pred)
