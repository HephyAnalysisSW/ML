vars = ['mva_Z1_eta',
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


if __name__ == "__main__":
    model = load_model("TTZ_TWZ_WZ_Keras_Model.h5")
    List = [[2.2721688747406006, -10.0, -10.0, -1.0, 263.69891357421875, -0.9794921875,
            -0.03838849067687988,
            0.0, -1.0, -1.0, 0.0, 72.53070831298828, -10.0, -1.0, 7.966670989990234,
            2.1394686698913574, -10.0, 0.0, -1.0, 0.0, 0.0, -1.0, -10.0, -10.0,
            -1.0, 3.892380475997925, 93.152099609375, -1.0, 0.0, 39.12520980834961]]
    pred = model.predict(List)
    print(pred)