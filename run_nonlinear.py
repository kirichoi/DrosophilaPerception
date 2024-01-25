# -*- coding: utf-8 -*-
#
# Kiri Choi, Won Kyu Kim, Changbong Hyeon
# School of Computational Sciences, Korea Institute for Advanced Study, 
# Seoul 02455, Korea
#
# This script reproduces figures in the manuscript titled
#
# Unveiling the Odor Representation in the Inner Brain of Drosophila through Compressed Sensing
#
# Figures related to testing various nonlinear filters are available in this
# Python script. Certain computation can take a long time and pre-computed array 
# files are available.
# Disable the flag to re-run the computation.
# CAUTION! - THIS CAN TAKE A LONG TIME!
# To view the figures, using the pickled files are highly recommended.
# 
# FLAG TO LOAD PRE-COMPUTED FILES #############################################
LOAD = True
###############################################################################

#%% Load datasets

import os
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy import stats
import pandas as pd
from neuprint.utils import connection_table_to_matrix
import matplotlib.pyplot as plt
import copy

np.random.seed(1234)

os.chdir(os.path.dirname(__file__))

FAFB_glo_info = pd.read_csv('./1-s2.0-S0960982220308587-mmc4.csv')

PNKC_df = pd.read_pickle(r'./data/PNKC_df.pkl')

PNKCbid = list(PNKC_df['bodyId'])
PNKCinstance = list(PNKC_df['instance'])
PNKCtype = list(PNKC_df['type'])

glo_labelKC_old = [s.split('_')[0] for s in PNKCtype]
glo_labelKC_old = [i.split('+') for i in glo_labelKC_old]

for j,i in enumerate(glo_labelKC_old):
    if len(i) > 1 and i[1] != '':
        neuprintId = PNKCbid[j]
        ugloidx = np.where(FAFB_glo_info['hemibrain_bodyid'] == neuprintId)[0]
        uglo = FAFB_glo_info['top_glomerulus'][ugloidx]
        glo_labelKC_old[j] = [uglo.iloc[0]]
    if len(i) > 1 and i[1] == '':
        glo_labelKC_old[j] = [i[0]]

glo_labelKC_old = np.array([item for sublist in glo_labelKC_old for item in sublist])

glo_labelKC = copy.deepcopy(glo_labelKC_old)

vc3m = np.where(glo_labelKC == 'VC3m')
vc3l = np.where(glo_labelKC == 'VC3l')
vc5 = np.where(glo_labelKC == 'VC5')

glo_labelKC[vc3m] = 'VC5'
glo_labelKC[vc3l] = 'VC3'
glo_labelKC[vc5] = 'VM6'

neuron_PNKC_df = pd.read_pickle(r'./data/neuron_PNKC_df.pkl')
conn_PNKC_df = pd.read_pickle(r'./data/conn_PNKC_df.pkl')
neuron_MBON_df = pd.read_pickle(r'./data/neuron_MBON_df3.pkl')
conn_MBON_df = pd.read_pickle(r'./data/conn_MBON_df3.pkl')

matrix_KC = connection_table_to_matrix(conn_PNKC_df, 'bodyId')
matrix_MBON = connection_table_to_matrix(conn_MBON_df, 'bodyId')

hallem_odor_sensitivity_raw = pd.read_excel('./data/Hallem_and_Carlson_2006_TS1.xlsx')
hallem_odor_sensitivity_raw = hallem_odor_sensitivity_raw.fillna(0)

hallem_odor_sensitivity = hallem_odor_sensitivity_raw.iloc[:,1:]
hallem_odor_type = hallem_odor_sensitivity_raw['Odor']
hallem_odor_sensitivity.index = hallem_odor_type
hallem_PN_type = hallem_odor_sensitivity_raw.columns[1:]

seki_odor_sensitivity_raw = pd.read_excel('./12915_2017_389_MOESM3_ESM.xlsx')
seki_odor_sensitivity_raw_t = seki_odor_sensitivity_raw.iloc[5:22, 1:]
colnames = list(seki_odor_sensitivity_raw.iloc[4][2:])
colnames.insert(0, 'Odor')
seki_odor_sensitivity_raw_t.columns = colnames
seki_odor_sensitivity_raw = seki_odor_sensitivity_raw_t.fillna(0)

seki_odor_sensitivity = seki_odor_sensitivity_raw.iloc[:,1:]
seki_odor_type = seki_odor_sensitivity_raw['Odor']
seki_odor_sensitivity.index = seki_odor_type
seki_PN_type = seki_odor_sensitivity_raw.columns[1:]

master_PN_type = np.unique(np.append(hallem_PN_type, seki_PN_type))
master_odor_type = np.insert(np.array(hallem_odor_type), 30, ['geosmin'])

master_odor_sensitivity = np.zeros((len(master_odor_type)+1, len(master_PN_type)+1), dtype=object)
master_odor_sensitivity[:,0][1:] = master_odor_type
master_odor_sensitivity[0][1:] = master_PN_type

for i in range(len(hallem_odor_sensitivity_raw)):
    row = hallem_odor_sensitivity_raw.iloc[i]
    odor_idx = np.where(row[0]==master_odor_type)[0][0]+1
    for j in range(len(row)-1):
        PN_idx = np.where(hallem_PN_type[j]==master_PN_type)[0][0]+1
        master_odor_sensitivity[odor_idx][PN_idx] = row[j+1]
    
for i in range(len(seki_odor_sensitivity_raw)):
    row = seki_odor_sensitivity_raw.iloc[i]
    odor_idx = np.where(row[0]==master_odor_type)[0][0]+1
    for j in range(len(row)-1):
        PN_idx = np.where(seki_PN_type[j]==master_PN_type)[0][0]+1
        master_odor_sensitivity[odor_idx][PN_idx] = row[j+1]
    
master_odor_sensitivity_array = master_odor_sensitivity[1:,1:].astype(float)

DM35idx = np.where(master_PN_type == 'DM35')[0][0]
DM3idx = np.where(master_PN_type == 'DM3')[0][0]
DM5idx = np.where(master_PN_type == 'DM5')[0][0]

master_odor_sensitivity_array[:,DM3idx] = master_odor_sensitivity_array[:,DM3idx] + master_odor_sensitivity_array[:,DM35idx]
master_odor_sensitivity_array[:,DM5idx] = master_odor_sensitivity_array[:,DM5idx] + master_odor_sensitivity_array[:,DM35idx]

master_odor_sensitivity_array = np.delete(master_odor_sensitivity_array, DM35idx, axis=1)
master_PN_type = np.delete(master_PN_type, DM35idx)

good_idx = []

for i,j in enumerate(master_odor_sensitivity_array):
    if len(np.where(np.abs(j) >= 40)[0]) != 0:
        good_idx.append(i)

master_odor_type = master_odor_type[good_idx]
master_odor_sensitivity_array = master_odor_sensitivity_array[good_idx]

master_odor_sensitivity_df = pd.DataFrame(master_odor_sensitivity_array)
master_odor_sensitivity_df.columns = master_PN_type
master_odor_sensitivity_df.index = master_odor_type

def L1norm(x):
    return np.linalg.norm(x, ord=1)

#%% Rearrange the dataset

PNKCbid_idx = []

for i in matrix_KC.index.values:
    PNKCbid_idx.append(PNKCbid.index(i))
    
KC_newidx = np.argsort(np.array(glo_labelKC)[PNKCbid_idx])
KC_newidx_label = np.sort(np.array(glo_labelKC)[PNKCbid_idx])
KC_sorted = np.sort(matrix_KC.columns.values)
KC_sortedidx = np.argsort(matrix_KC.columns.values)

matrix_KC_re = np.array(matrix_KC)[KC_newidx][:,KC_sortedidx]

MBON_sortedidx = np.argsort(matrix_MBON.columns.values)

matrix_MBONKC = matrix_MBON.loc[KC_sorted]
matrix_MBONKC = np.array(matrix_MBONKC)[:,MBON_sortedidx]

matrix_MBONKCidx = np.nonzero(np.sum(matrix_MBONKC, axis=0))[0]

matrix_MBONKC = matrix_MBONKC[:,matrix_MBONKCidx]

odor_chemtype_dict = {'ammonium hydroxide': 'amines', 'putrescine' : 'amines',  
                      'cadaverine': 'amines', 'g-butyrolactone': 'lactones', 
                      'g-hexalactone': 'lactones', 'g-octalactone': 'lactones', 
                      'g-decalactone': 'lactones','d-decalactone': 'lactones', 
                      'methanoic acid': 'acids', 'acetic acid': 'acids', 
                      'propionic acid': 'acids', 'butyric acid': 'acids', 
                      'pentanoic acid': 'acids', 'hexanoic acid': 'acids', 
                      'heptanoic acid': 'acids', 'octanoic acid': 'acids', 
                      'nonanoic acid': 'acids', 'linoleic acid': 'acids', 
                      'isobutyric acid': 'acids', 'isopentanoic acid': 'acids', 
                      'pyruvic acid': 'acids', '2-ethylhexanoic acid': 'acids', 
                      'lactic acid': 'acids', '3-methylthio-1-propanol': 'sulfur compounds', 
                      'dimethyl sulfide': 'sulfur compounds', 'terpinolene': 'terpenes',
                      'a -pinene': 'terpenes', 'b -pinene': 'terpenes', 
                      '(1S)-(+)-3-carene': 'terpenes', 'limonene': 'terpenes',
                      'a-humulene': 'terpenes', 'b -myrcene': 'terpenes', 
                      '(-)-trans-caryophyllene': 'terpenes', 'p-cymene': 'terpenes', 
                      'geranyl acetate': 'terpenes', 'a -terpineol': 'terpenes', 
                      'geraniol': 'terpenes', 'nerol': 'terpenes', 
                      'linalool': 'terpenes', 'b-citronellol': 'terpenes', 
                      'linalool oxide': 'terpenes', 'acetaldehyde': 'aldehydes', 
                      'propanal': 'aldehydes', 'butanal': 'aldehydes', 
                      'pentanal': 'aldehydes', 'hexanal': 'aldehydes', 
                      'E2-hexenal': 'aldehydes', 'furfural': 'aldehydes', 
                      '2-propenal': 'aldehydes', 'acetone': 'ketones', 
                      '2-butanone': 'ketones', '2-pentanone': 'ketones', 
                      '2-heptanone': 'ketones', '6-methyl-5-hepten-2-one': 'ketones', 
                      '2,3-butanedione': 'ketones', 'phenethyl alcohol': 'aromatics',
                      'benzyl alcohol': 'aromatics', 'methyl salicylate': 'aromatics', 
                      'methyl benzoate': 'aromatics', 'ethyl benzoate': 'aromatics', 
                      'phenethyl acetate': 'aromatics', 'benzaldehyde': 'aromatics',
                      'phenylacetaldehyde': 'aromatics', 'acetophenone': 'aromatics', 
                      'ethyl cinnamate': 'aromatics', '2-methylphenol': 'aromatics',
                      '4-ethyl guaiacol': 'aromatics', 'eugenol': 'aromatics', 
                      'methanol': 'alcohols', 'ethanol': 'alcohols', 
                      '1-propanol': 'alcohols', '1-butanol': 'alcohols', 
                      '1-pentanol': 'alcohols', '1-hexanol': 'alcohols', 
                      '1-octanol': 'alcohols', '2-pentanol': 'alcohols',
                      '3-methylbutanol': 'alcohols', '3-methyl-2-buten-1-ol': 'alcohols',
                      '1-penten-3-ol': 'alcohols', '1-octen-3-ol': 'alcohols', 
                      'E2-hexenol': 'alcohols', 'Z2-hexenol': 'alcohols', 
                      'E3-hexenol': 'alcohols', 'Z3-hexenol': 'alcohols', 
                      'glycerol': 'alcohols', '2,3-butanediol': 'alcohols',
                      'methyl acetate': 'esters', 'ethyl acetate': 'esters', 
                      'propyl acetate': 'esters', 'butyl acetate': 'esters', 
                      'pentyl acetate': 'esters', 'hexyl acetate': 'esters', 
                      'isobutyl acetate': 'esters', 'isopentyl acetate': 'esters', 
                      'E2-hexenyl acetate': 'esters', 'methyl butyrate': 'esters', 
                      'ethyl butyrate': 'esters', 'hexyl butyrate': 'esters',
                      'ethyl 3-hydroxybutyrate': 'esters', 'ethyl propionate': 'esters', 
                      'ethyl methanoate': 'esters', 'methyl hexanoate': 'esters', 
                      'ethyl hexanoate': 'esters', 'hexyl hexanoate': 'esters',
                      'methyl octanoate': 'esters', 'ethyl octanoate': 'esters',
                      'ethyl decanoate': 'esters', 'ethyl trans-2-butenoate': 'esters', 
                      'ethyl lactate': 'esters', 'diethyl succinate': 'esters',
                      'geosmin': 'terpenes', 'spontaneous firing rate': 'other'}

alltarglo = np.unique(KC_newidx_label)

odor_glo = []

for odor in master_odor_type:
    spike = master_odor_sensitivity_df.loc[odor].to_numpy()
    gidx = np.where(np.abs(spike) >= 51)[0]
    odor_glo.append(list(master_PN_type[gidx]))

Smat = np.zeros(np.shape(matrix_MBONKC.T))

for i in range(len(matrix_MBONKC.T)):
    Smat[i] = matrix_MBONKC.T[i]/np.linalg.norm(matrix_MBONKC.T[i])

Psimat = np.zeros(np.shape(matrix_KC_re.T))

for i in range(len(matrix_KC_re)):
    Psimat[:,i] = matrix_KC_re[i]/np.linalg.norm(matrix_KC_re[i])

Theta = np.matmul(Smat, Psimat)

master_odor_color = []

for o in master_odor_type:
    if odor_chemtype_dict[o] == 'amines':
        master_odor_color.append('#a01315')
    elif odor_chemtype_dict[o] == 'lactones':
        master_odor_color.append('#f62124')
    elif odor_chemtype_dict[o] == 'acids':
        master_odor_color.append('#fe4a91')
    elif odor_chemtype_dict[o] == 'sulfur compounds':
        master_odor_color.append('#5602ac')
    elif odor_chemtype_dict[o] == 'terpenes':
        master_odor_color.append('#ac5ffa')
    elif odor_chemtype_dict[o] == 'aldehydes':
        master_odor_color.append('#153ac9')
    elif odor_chemtype_dict[o] == 'ketones':
        master_odor_color.append('#0f9af0')
    elif odor_chemtype_dict[o] == 'aromatics':
        master_odor_color.append('#0ff0ae')
    elif odor_chemtype_dict[o] == 'alcohols':
        master_odor_color.append('#1fb254')
    elif odor_chemtype_dict[o] == 'esters':
        master_odor_color.append('#60b82c')
    else:
        master_odor_color.append('#000000')

singleinput = []
allsingletruPNactivity = []

for o in master_odor_type:
    spike = copy.deepcopy(master_odor_sensitivity_df.loc[o].to_numpy())
    spike[np.abs(spike) < 40] = 0
    
    truPNactivity = np.zeros(len(KC_newidx_label))
    
    for i in range(len(alltarglo)):
        gloidx = np.where(KC_newidx_label == alltarglo[i])[0]
        if alltarglo[i] in master_PN_type:
            s = np.where(alltarglo[i] == master_PN_type)[0][0]
            truPNactivity[gloidx] = spike[s]
        
    allsingletruPNactivity.append(truPNactivity) 
    
    KCact = np.dot(Psimat, truPNactivity)
    
    y = np.dot(Smat, KCact)
    singleinput.append(y)

#%% Nonlinear filters - ReLU

matrix_MBONKC_ReLU = copy.deepcopy(matrix_MBONKC)

Smat_ReLU = np.zeros(np.shape(matrix_MBONKC_ReLU.T))

for i in range(len(matrix_MBONKC_ReLU.T)):
    Smat_ReLU[i] = matrix_MBONKC_ReLU.T[i]/np.linalg.norm(matrix_MBONKC_ReLU.T[i])

matrix_KC_re_ReLU = copy.deepcopy(matrix_KC_re)
matrix_KC_re_ReLU = matrix_KC_re_ReLU*(np.abs(matrix_KC_re_ReLU)>7)

matrix_KC_re_ReLU = np.delete(matrix_KC_re_ReLU, np.where(np.sum(matrix_KC_re_ReLU, axis=1) == 0), axis=0)

Psimat_ReLU = np.zeros(np.shape(matrix_KC_re_ReLU.T))

for i in range(len(matrix_KC_re_ReLU)):
    Psimat_ReLU[:,i] = matrix_KC_re_ReLU[i]/np.linalg.norm(matrix_KC_re_ReLU[i])

Theta_ReLU = np.matmul(Smat_ReLU, Psimat_ReLU)

singleinput_ReLU = []
allsingletruPNactivity_ReLU = []

for o in master_odor_type:
    spike = copy.deepcopy(master_odor_sensitivity_df.loc[o].to_numpy())
    spike[np.abs(spike) < 40] = 0
    
    truPNactivity = np.zeros(len(KC_newidx_label)-4)
    
    for i in range(len(alltarglo)):
        gloidx = np.where(KC_newidx_label == alltarglo[i])[0]
        if alltarglo[i] in master_PN_type:
            s = np.where(alltarglo[i] == master_PN_type)[0][0]
            truPNactivity[gloidx] = spike[s]
        
    allsingletruPNactivity_ReLU.append(truPNactivity)
    
    KCact = np.dot(Psimat_ReLU, truPNactivity)
    y = np.dot(Smat_ReLU, KCact)
    
    singleinput_ReLU.append(y)

#%% Residual calculation for single odorants using ReLU

if not LOAD:
    np.random.seed(1234)
    
    allsingleres_ReLU = []
    
    for i,o in enumerate(master_odor_type):
        print(o)
        
        truPNactivity = allsingletruPNactivity_ReLU[i]
        
        y = singleinput_ReLU[i]
        
        bounds = scipy.optimize.Bounds(lb=-np.inf, ub=np.inf)
        constr = ({'type': 'eq', 'fun': lambda x: Theta_ReLU @ x - y})
        
        x0 = np.linalg.pinv(Theta_ReLU) @ y
        
        res = minimize(L1norm, x0, method='SLSQP', bounds=bounds, constraints=constr, options={'maxiter': 10000})
        
        allsingleres_ReLU.append(res.x)
        
    single_residuals_ReLU = []
    
    for l,i in enumerate(allsingleres_ReLU):
        r_temp = []
        for k,j in enumerate(singleinput_ReLU):
            gidx = np.where(np.abs(allsingletruPNactivity_ReLU[k]) >= 40)[0]
            KCpred = np.dot(Psimat_ReLU[:,gidx], i[gidx])
            r = np.linalg.norm(j-np.dot(Smat_ReLU,KCpred))/np.linalg.norm(j)
            r_temp.append(r)
        single_residuals_ReLU.append(r_temp)
else:
    single_residuals_ReLU = np.load('./precalc/single_residuals_ReLU_3.npy')

#%% Nonlinear filters - SNIC

SNIC_response = np.load(r'./data/SNIC_resp.npy')

SNIC_input_int = SNIC_response[0]
SNIC_response_int = SNIC_response[1]/10

matrix_MBONKC_SNIC = copy.deepcopy(matrix_MBONKC)

Smat_SNIC = np.zeros(np.shape(matrix_MBONKC_SNIC.T))

for i in range(len(matrix_MBONKC_SNIC.T)):
    Smat_SNIC[i] = matrix_MBONKC_SNIC.T[i]/np.linalg.norm(matrix_MBONKC_SNIC.T[i])

matrix_KC_re_SNIC = copy.deepcopy(matrix_KC_re)
matrix_KC_re_SNIC = np.interp(matrix_KC_re_SNIC, SNIC_response[0], SNIC_response[1]/10)

matrix_KC_re_SNIC = np.delete(matrix_KC_re_SNIC, np.where(np.sum(matrix_KC_re_SNIC, axis=1) == 0), axis=0)

Psimat_SNIC = np.zeros(np.shape(matrix_KC_re_SNIC.T))

for i in range(len(matrix_KC_re_SNIC)):
    Psimat_SNIC[:,i] = matrix_KC_re_SNIC[i]/np.linalg.norm(matrix_KC_re_SNIC[i])

Theta_SNIC = np.matmul(Smat_SNIC, Psimat_SNIC)

singleinput_SNIC = []
allsingletruPNactivity_SNIC = []

for o in master_odor_type:
    spike = copy.deepcopy(master_odor_sensitivity_df.loc[o].to_numpy())
    spike[np.abs(spike) < 40] = 0
    
    truPNactivity = np.zeros(len(KC_newidx_label)-4)
    
    for i in range(len(alltarglo)):
        gloidx = np.where(KC_newidx_label == alltarglo[i])[0]
        if alltarglo[i] in master_PN_type:
            s = np.where(alltarglo[i] == master_PN_type)[0][0]
            truPNactivity[gloidx] = spike[s]
        
    allsingletruPNactivity_SNIC.append(truPNactivity)
    
    KCact = np.dot(Psimat_SNIC, truPNactivity)
    y = np.dot(Smat_SNIC, KCact)
    
    singleinput_SNIC.append(y)

#%% Residual calculation for single odorants using SNIC

if not LOAD:
    np.random.seed(1234)
    
    allsingleres_SNIC = []
    
    for i,o in enumerate(master_odor_type):
        print(o)
        
        truPNactivity = allsingletruPNactivity_SNIC[i]
        
        y = singleinput_SNIC[i]
        
        bounds = scipy.optimize.Bounds(lb=-np.inf, ub=np.inf)
        constr = ({'type': 'eq', 'fun': lambda x: Theta_SNIC @ x - y})
        
        x0 = np.linalg.pinv(Theta_SNIC) @ y
        
        res = minimize(L1norm, x0, method='SLSQP', bounds=bounds, constraints=constr, options={'maxiter': 10000})
        
        allsingleres_SNIC.append(res.x)
        
    single_residuals_SNIC = []
    
    for l,i in enumerate(allsingleres_SNIC):
        r_temp = []
        for k,j in enumerate(singleinput_SNIC):
            gidx = np.where(np.abs(allsingletruPNactivity_SNIC[k]) >= 40)[0]
            KCpred = np.dot(Psimat_SNIC[:,gidx], i[gidx])
            r = np.linalg.norm(j-np.dot(Smat_SNIC,KCpred))/np.linalg.norm(j)
            r_temp.append(r)
        single_residuals_SNIC.append(r_temp)
else:
    single_residuals_SNIC = np.load('./precalc/single_residuals_SNIC_3.npy')

#%% Residual calculation for single odorants using sigmoid

def sigmoid(x):
    return 1/(1 + np.exp(-1*(x-3)))

matrix_MBONKC_sigmoid = copy.deepcopy(matrix_MBONKC)

Smat_sigmoid = np.zeros(np.shape(matrix_MBONKC_sigmoid.T))

for i in range(len(matrix_MBONKC_sigmoid.T)):
    Smat_sigmoid[i] = matrix_MBONKC_sigmoid.T[i]/np.linalg.norm(matrix_MBONKC_sigmoid.T[i])

matrix_KC_re_sigmoid = copy.deepcopy(matrix_KC_re)
matrix_KC_re_sigmoid = sigmoid(matrix_KC_re_sigmoid)

matrix_KC_re_sigmoid = np.delete(matrix_KC_re_sigmoid, np.where(np.sum(matrix_KC_re_sigmoid, axis=1) == 0), axis=0)

Psimat_sigmoid = np.zeros(np.shape(matrix_KC_re_sigmoid.T))

for i in range(len(matrix_KC_re_sigmoid)):
    Psimat_sigmoid[:,i] = matrix_KC_re_sigmoid[i]/np.linalg.norm(matrix_KC_re_sigmoid[i])

Theta_sigmoid = np.matmul(Smat_sigmoid, Psimat_sigmoid)

singleinput_sigmoid = []
allsingletruPNactivity_sigmoid = []

for o in master_odor_type:
    spike = copy.deepcopy(master_odor_sensitivity_df.loc[o].to_numpy())
    spike[np.abs(spike) < 40] = 0
    
    truPNactivity = np.zeros(len(KC_newidx_label))
    
    for i in range(len(alltarglo)):
        gloidx = np.where(KC_newidx_label == alltarglo[i])[0]
        if alltarglo[i] in master_PN_type:
            s = np.where(alltarglo[i] == master_PN_type)[0][0]
            truPNactivity[gloidx] = spike[s]
        
    allsingletruPNactivity_sigmoid.append(truPNactivity)
    
    KCact = np.dot(Psimat_sigmoid, truPNactivity)
    y = np.dot(Smat_sigmoid, KCact)
    
    singleinput_sigmoid.append(y)

#%% Residual calculation for single odorants using sigmoid

if not LOAD:
    np.random.seed(1234)
    
    allsingleres_sigmoid = []
    
    for i,o in enumerate(master_odor_type):
        print(o)
        
        truPNactivity = allsingletruPNactivity_sigmoid[i]
        
        y = singleinput_sigmoid[i]
        
        bounds = scipy.optimize.Bounds(lb=-np.inf, ub=np.inf)
        constr = ({'type': 'eq', 'fun': lambda x: Theta_sigmoid @ x - y})
        
        x0 = np.linalg.pinv(Theta_sigmoid) @ y
        
        res = minimize(L1norm, x0, method='SLSQP', bounds=bounds, constraints=constr, options={'maxiter': 10000})
        
        allsingleres_sigmoid.append(res.x)
        
    single_residuals_sigmoid = []
    
    for l,i in enumerate(allsingleres_sigmoid):
        r_temp = []
        for k,j in enumerate(singleinput_sigmoid):
            gidx = np.where(np.abs(allsingletruPNactivity_sigmoid[k]) >= 40)[0]
            KCpred = np.dot(Psimat_sigmoid[:,gidx], i[gidx])
            r = np.linalg.norm(j-np.dot(Smat_sigmoid,KCpred))/np.linalg.norm(j)
            r_temp.append(r)
        single_residuals_sigmoid.append(r_temp)
else:
    single_residuals_sigmoid = np.load('./precalc/single_residuals_sigmoid_3.npy')

#%% Figure 9A - Z-scores for linear vs nonlinear

single_residuals = np.load('./precalc/single_residuals3.npy')
zscores = -scipy.stats.zscore(single_residuals, axis=1)

zscores_ReLU = -scipy.stats.zscore(single_residuals_ReLU, axis=1)
zscores_SNIC = -scipy.stats.zscore(single_residuals_SNIC, axis=1)
zscores_sigmoid = -scipy.stats.zscore(single_residuals_sigmoid, axis=1)

ps_ReLU, pval_ReLU = stats.pearsonr(np.diag(zscores), np.diag(zscores_ReLU))
ps_SNIC, pval_SNIC = stats.pearsonr(np.diag(zscores), np.diag(zscores_SNIC))
ps_sigmoid, pval_sigmoid = stats.pearsonr(np.diag(zscores), np.diag(zscores_sigmoid))

fig, ax = plt.subplots(figsize=(4,3.5))
plt.scatter(np.diag(zscores), np.diag(zscores_ReLU), facecolors='none', edgecolor='tab:blue')
plt.scatter(np.diag(zscores), np.diag(zscores_SNIC), facecolors='none', edgecolor='tab:red')
plt.scatter(np.diag(zscores), np.diag(zscores_sigmoid), facecolors='none', edgecolor='tab:green')
plt.text(10, 0.15, r'$\rho={0}, p\ll$'.format(np.around(ps_ReLU, 3))+r'$10^{-8}$', 
         fontsize=13, color='tab:blue')
plt.text(10, 0.23, r'$\rho={0}, p\ll$'.format(np.around(ps_SNIC, 3))+r'$10^{-8}$', 
         fontsize=13, color='tab:red')
plt.text(10, 0.36, r'$\rho={0}, p\ll$'.format(np.around(ps_sigmoid, 3))+r'$10^{-8}$', 
         fontsize=13, color='tab:green')
plt.xscale('log')
plt.yscale('log')
plt.xticks([1, 10], ['$-10^{0}$', '$-10^{1}$'], fontsize=15)
plt.xlabel('$Z$-score, Linear', fontsize=15)
plt.yticks([0.1, 1, 10], ['$-10^{-1}$', '$-10^{0}$', '$-10^{1}$'], fontsize=15)
plt.ylabel('$Z$-score, Nonlinear', fontsize=15)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.legend(['ReLU', 'SNIC', 'Sigmoid'], fontsize=13, loc=4)
plt.show()

#%% Figure 9B - self residuals for linear vs nonlinear

ps_ReLU, pval_ReLU = stats.pearsonr(np.diag(single_residuals), np.diag(single_residuals_ReLU))
ps_SNIC, pval_SNIC = stats.pearsonr(np.diag(single_residuals), np.diag(single_residuals_SNIC))
ps_sigmoid, pval_sigmoid = stats.pearsonr(np.diag(single_residuals), np.diag(single_residuals_sigmoid))

fig, ax = plt.subplots(figsize=(4,3.5))
plt.scatter(np.diag(single_residuals), np.diag(single_residuals_ReLU), facecolors='none', edgecolor='tab:blue')
plt.scatter(np.diag(single_residuals), np.diag(single_residuals_SNIC), facecolors='none', edgecolor='tab:red')
plt.scatter(np.diag(single_residuals), np.diag(single_residuals_sigmoid), facecolors='none', edgecolor='tab:green')
plt.xscale('log')
plt.yscale('log')
plt.xticks([1e-9, 1e-7, 1e-5, 1e-3, 1e-1], fontsize=15)
plt.xlabel(r'$r_{\alpha|\alpha}$, Linear', fontsize=15)
plt.yticks([1e-9, 1e-7, 1e-5, 1e-3, 1e-1], fontsize=15)
plt.ylabel(r'$r_{\alpha|\alpha}$, Nonlinear', fontsize=15)
plt.show()



