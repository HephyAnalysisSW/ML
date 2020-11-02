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

treename = 'Events'
filename = {}


# key is used as name on the plot
filename['TWZ'] = '/local/mmoser/root_files/TWZ_NLO_DR.root'
filename['TTZ'] = '/local/mmoser/root_files/TTZ.root'
filename['WZ'] = '/local/mmoser/root_files/WZ.root'
filename['NON'] = '/local/mmoser/root_files/nonprompt_3l.root'

# model savepath:
model_path = '/local/mmoser/'

# for the plots
save_path = '/local/mmoser/plots/'

NDIM = len(variables)

# Network layout:
NL = [NDIM*5, NDIM*5, NDIM*5]

make_plots = True # Histogram plots
make_roc = True # ROC curve plot
make_conf = True # print confusion matrix
