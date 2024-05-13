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
# Figures related to testing and comparing random connectivity matrices are 
# available in this Python script.
# Certain computation can take a long time and pre-computed array files are
# available.
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

Smat = np.zeros(np.shape(matrix_MBONKC.T))

for i in range(len(matrix_MBONKC.T)):
    Smat[i] = matrix_MBONKC.T[i]/np.linalg.norm(matrix_MBONKC.T[i])

Psimat = np.zeros(np.shape(matrix_KC_re.T))

for i in range(len(matrix_KC_re)):
    Psimat[:,i] = matrix_KC_re[i]/np.linalg.norm(matrix_KC_re[i])

Theta = np.matmul(Smat, Psimat)

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

single_residuals = np.load('./precalc/single_residuals3.npy')

zscores = -scipy.stats.zscore(single_residuals, axis=1)

#%% Random matrices

matrix_PNKC_rsparse = scipy.sparse.random(np.shape(matrix_KC_re)[0], np.shape(matrix_KC_re)[1], density=0.0543).toarray()
matrix_PNKC_rgaussian = np.random.normal(size=(np.shape(matrix_KC_re)[0], np.shape(matrix_KC_re)[1]))
matrix_PNKC_rgaussian[matrix_PNKC_rgaussian<0] = 0 # negative connectivity does not make sense
matrix_PNKC_rbernoulli = np.random.binomial(1, 0.0543, size=(np.shape(matrix_KC_re)[0], np.shape(matrix_KC_re)[1]))
dp = copy.deepcopy(matrix_KC_re)
dpf = dp.flatten()
np.random.shuffle(dpf)
matrix_PNKC_shuffled = np.reshape(dpf, np.shape(matrix_KC_re))

matrix_KCMBON_rsparse = scipy.sparse.random(np.shape(matrix_MBONKC)[0], np.shape(matrix_MBONKC)[1], density=0.286).toarray()
matrix_KCMBON_rgaussian = np.random.normal(size=(np.shape(matrix_MBONKC)[0], np.shape(matrix_MBONKC)[1]))
matrix_KCMBON_rgaussian[matrix_KCMBON_rgaussian<0] = 0 # negative connectivity does not make sense
matrix_KCMBON_rbernoulli = np.random.binomial(1, 0.286, size=(np.shape(matrix_MBONKC)[0], np.shape(matrix_MBONKC)[1]))
dp = copy.deepcopy(matrix_MBONKC)
dpf = dp.flatten()
np.random.shuffle(dpf)
matrix_KCMBON_shuffled = np.reshape(dpf, np.shape(matrix_MBONKC))

#%% Sparse random residual calculation

Smat_sparse = np.zeros(np.shape(matrix_MBONKC.T))

for i in range(len(matrix_MBONKC.T)):
    Smat_sparse[i] = matrix_KCMBON_rsparse.T[i]/np.linalg.norm(matrix_KCMBON_rsparse.T[i])

Psimat_sparse = np.zeros(np.shape(matrix_KC_re.T))

for i in range(len(matrix_KC_re)):
    Psimat_sparse[:,i] = matrix_PNKC_rsparse[i]/np.linalg.norm(matrix_PNKC_rsparse[i])

Theta_sparse = np.matmul(Smat_sparse, Psimat_sparse)

singleinput_sparse = []

for i,o in enumerate(master_odor_type):
    KCact = np.dot(Psimat_sparse, allsingletruPNactivity[i])
    
    y = np.dot(Smat_sparse, KCact)
    singleinput_sparse.append(y)

if not LOAD:
    np.random.seed(1234)
    
    allsingleres_sparse = []
    
    for i,o in enumerate(master_odor_type):
        print(o)
        
        truPNactivity = allsingletruPNactivity[i]
        
        y = singleinput_sparse[i]
        
        bounds = scipy.optimize.Bounds(lb=-np.inf, ub=np.inf)
        constr = ({'type': 'eq', 'fun': lambda x: Theta_sparse @ x - y})
        
        x0 = np.linalg.pinv(Theta_sparse) @ y
        
        res = minimize(L1norm, x0, method='SLSQP', bounds=bounds, constraints=constr, options={'maxiter': 10000})
        
        allsingleres_sparse.append(res.x)
        
    single_residuals_sparse = []
        
    for l,i in enumerate(allsingleres_sparse):
        r_temp = []
        for k,j in enumerate(singleinput_sparse):
            gidx = np.where(np.abs(allsingletruPNactivity[k]) >= 40)[0]
            r_temp.append(np.linalg.norm(j-np.dot(Smat_sparse,np.dot(Psimat_sparse[:,gidx], i[gidx])))/np.linalg.norm(j))
        single_residuals_sparse.append(r_temp)
else:
    single_residuals_sparse = np.load('./precalc/single_residuals_sparse_sampling_1.npy')

zscores_sparse = -scipy.stats.zscore(single_residuals_sparse, axis=1)

#%% Gaussian random residual calculation

Smat_gaussian = np.zeros(np.shape(matrix_MBONKC.T))

for i in range(len(matrix_MBONKC.T)):
    Smat_gaussian[i] = matrix_KCMBON_rgaussian.T[i]/np.linalg.norm(matrix_KCMBON_rgaussian.T[i])

Psimat_gaussian = np.zeros(np.shape(matrix_KC_re.T))

for i in range(len(matrix_KC_re)):
    Psimat_gaussian[:,i] = matrix_PNKC_rgaussian[i]/np.linalg.norm(matrix_PNKC_rgaussian[i])

Theta_gaussian = np.matmul(Smat_gaussian, Psimat_gaussian)

singleinput_gaussian = []

for i,o in enumerate(master_odor_type):
    KCact = np.dot(Psimat_gaussian, allsingletruPNactivity[i])
    
    y = np.dot(Smat_gaussian, KCact)
    singleinput_gaussian.append(y)

if not LOAD:
    np.random.seed(1234)
    
    allsingleres_gaussian = []
    
    for i,o in enumerate(master_odor_type):
        print(o)
        
        truPNactivity = allsingletruPNactivity[i]
        
        y = singleinput_gaussian[i]
        
        bounds = scipy.optimize.Bounds(lb=-np.inf, ub=np.inf)
        constr = ({'type': 'eq', 'fun': lambda x: Theta_gaussian @ x - y})
        
        x0 = np.linalg.pinv(Theta_gaussian) @ y
        
        res = minimize(L1norm, x0, method='SLSQP', bounds=bounds, constraints=constr, options={'maxiter': 10000})
        
        allsingleres_gaussian.append(res.x)
        
    single_residuals_gaussian = []
        
    for l,i in enumerate(allsingleres_gaussian):
        r_temp = []
        for k,j in enumerate(singleinput_gaussian):
            gidx = np.where(np.abs(allsingletruPNactivity[k]) >= 40)[0]
            r_temp.append(np.linalg.norm(j-np.dot(Smat_gaussian,np.dot(Psimat_gaussian[:,gidx], i[gidx])))/np.linalg.norm(j))
        single_residuals_gaussian.append(r_temp)
else:
    single_residuals_gaussian = np.load('./precalc/single_residuals_gaussian_sampling_1.npy')

zscores_gaussian = -scipy.stats.zscore(single_residuals_gaussian, axis=1)

#%% Bernoulli random residual calculation

Smat_bernoulli = np.zeros(np.shape(matrix_MBONKC.T))

for i in range(len(matrix_MBONKC.T)):
    Smat_bernoulli[i] = matrix_KCMBON_rbernoulli.T[i]/np.linalg.norm(matrix_KCMBON_rbernoulli.T[i])

Psimat_bernoulli = np.zeros(np.shape(matrix_KC_re.T))

for i in range(len(matrix_KC_re)):
    Psimat_bernoulli[:,i] = matrix_PNKC_rbernoulli[i]/np.linalg.norm(matrix_PNKC_rbernoulli[i])

Theta_bernoulli = np.matmul(Smat_bernoulli, Psimat_bernoulli)

singleinput_bernoulli = []

for i,o in enumerate(master_odor_type):
    KCact = np.dot(Psimat_bernoulli, allsingletruPNactivity[i])
    
    y = np.dot(Smat_bernoulli, KCact)
    singleinput_bernoulli.append(y)

if not LOAD:
    np.random.seed(1234)
    
    allsingleres_bernoulli = []
    
    for i,o in enumerate(master_odor_type):
        print(o)
        
        truPNactivity = allsingletruPNactivity[i]
        
        y = singleinput_bernoulli[i]
        
        bounds = scipy.optimize.Bounds(lb=-np.inf, ub=np.inf)
        constr = ({'type': 'eq', 'fun': lambda x: Theta_bernoulli @ x - y})
        
        x0 = np.linalg.pinv(Theta_bernoulli) @ y
        
        res = minimize(L1norm, x0, method='SLSQP', bounds=bounds, constraints=constr, options={'maxiter': 10000})
        
        allsingleres_bernoulli.append(res.x)
        
    single_residuals_bernoulli = []
        
    for l,i in enumerate(allsingleres_bernoulli):
        r_temp = []
        for k,j in enumerate(singleinput_bernoulli):
            gidx = np.where(np.abs(allsingletruPNactivity[k]) >= 40)[0]
            r_temp.append(np.linalg.norm(j-np.dot(Smat_bernoulli,np.dot(Psimat_bernoulli[:,gidx], i[gidx])))/np.linalg.norm(j))
        single_residuals_bernoulli.append(r_temp)
else:
    single_residuals_bernoulli = np.load('./precalc/single_residuals_bernoulli_sampling_1.npy')

zscores_bernoulli = -scipy.stats.zscore(single_residuals_bernoulli, axis=1)

#%% Random shuffled residual calculation

Smat_shuffled = np.zeros(np.shape(matrix_MBONKC.T))

for i in range(len(matrix_MBONKC.T)):
    Smat_shuffled[i] = matrix_KCMBON_shuffled.T[i]/np.linalg.norm(matrix_KCMBON_shuffled.T[i])

Psimat_shuffled = np.zeros(np.shape(matrix_KC_re.T))

for i in range(len(matrix_KC_re)):
    Psimat_shuffled[:,i] = matrix_PNKC_shuffled[i]/np.linalg.norm(matrix_PNKC_shuffled[i])

Theta_shuffled = np.matmul(Smat_shuffled, Psimat_shuffled)

singleinput_shuffled = []

for i,o in enumerate(master_odor_type):
    KCact = np.dot(Psimat_shuffled, allsingletruPNactivity[i])
    
    y = np.dot(Smat_shuffled, KCact)
    singleinput_shuffled.append(y)

if not LOAD:
    np.random.seed(1234)
    
    allsingleres_shuffled = []
    
    for i,o in enumerate(master_odor_type):
        print(o)
        
        truPNactivity = allsingletruPNactivity[i]
        
        y = singleinput_shuffled[i]
        
        bounds = scipy.optimize.Bounds(lb=-np.inf, ub=np.inf)
        constr = ({'type': 'eq', 'fun': lambda x: Theta_shuffled @ x - y})
        
        x0 = np.linalg.pinv(Theta_shuffled) @ y
        
        res = minimize(L1norm, x0, method='SLSQP', bounds=bounds, constraints=constr, options={'maxiter': 10000})
        
        allsingleres_shuffled.append(res.x)
        
    single_residuals_shuffled = []
        
    for l,i in enumerate(allsingleres_shuffled):
        r_temp = []
        for k,j in enumerate(singleinput_shuffled):
            gidx = np.where(np.abs(allsingletruPNactivity[k]) >= 40)[0]
            r_temp.append(np.linalg.norm(j-np.dot(Smat_shuffled,np.dot(Psimat_shuffled[:,gidx], i[gidx])))/np.linalg.norm(j))
        single_residuals_shuffled.append(r_temp)
else:
    single_residuals_shuffled = np.load('./precalc/single_residuals_shuffled_sampling_1.npy')

zscores_shuffled = -scipy.stats.zscore(single_residuals_shuffled, axis=1)

#%% Fig 12 - Comparing the performance between random matrices and observed connectivity


fig = plt.figure(figsize=(6,3))
plt.boxplot([np.diag(zscores)[:-1], np.diag(zscores_sparse)[:-1],
             np.diag(zscores)[:-1], np.diag(zscores_bernoulli)[:-1],
             np.diag(zscores)[:-1], np.diag(zscores_shuffled)[:-1],
             np.diag(zscores)[:-1], np.diag(zscores_gaussian)[:-1]], 
            positions=[0,0.75,2,2.75,4,4.75,6,6.75],
            showfliers=False, zorder=100)
plt.scatter(np.repeat(0, len(zscores)-1), np.diag(zscores)[:-1], s=10, 
            edgecolors='none', alpha=0.5, facecolors='gray')
plt.scatter(np.repeat(2, len(zscores)-1), np.diag(zscores)[:-1], s=10, 
            edgecolors='none', alpha=0.5, facecolors='gray')
plt.scatter(np.repeat(4, len(zscores)-1), np.diag(zscores)[:-1], s=10, 
            edgecolors='none', alpha=0.5, facecolors='gray')
plt.scatter(np.repeat(6, len(zscores)-1), np.diag(zscores)[:-1], s=10, 
            edgecolors='none', alpha=0.5, facecolors='gray')
plt.scatter(np.repeat(0.75, len(zscores)-1), np.diag(zscores_sparse)[:-1], s=10, 
            edgecolors='none', alpha=0.5, facecolors='gray')
plt.scatter(np.repeat(2.75, len(zscores)-1), np.diag(zscores_bernoulli)[:-1], s=10, 
            edgecolors='none', alpha=0.5, facecolors='gray')
plt.scatter(np.repeat(4.75, len(zscores)-1), np.diag(zscores_shuffled)[:-1], s=10, 
            edgecolors='none', alpha=0.5, facecolors='gray')
plt.scatter(np.repeat(6.75, len(zscores)-1), np.diag(zscores_gaussian)[:-1], s=10, 
            edgecolors='none', alpha=0.5, facecolors='gray')
plt.xticks([0,0.75,2,2.75,4,4.75,6,6.75], ['Original', 'Sparse', 'Original', 'Bernoulli',
                                           'Original', 'Shuffled', 'Original', 'Gaussian'], 
           fontsize=12, rotation=35, ha='right')
for i in range(len(zscores)-1):
    plt.plot([0, 0.75], [np.diag(zscores)[i], np.diag(zscores_sparse)[i]], 
             c='gray', alpha=0.5, lw=0.5)
    plt.plot([2, 2.75], [np.diag(zscores)[i], np.diag(zscores_bernoulli)[i]], 
             c='gray', alpha=0.5, lw=0.5)
    plt.plot([4, 4.75], [np.diag(zscores)[i], np.diag(zscores_shuffled)[i]], 
             c='gray', alpha=0.5, lw=0.5)
    plt.plot([6, 6.75], [np.diag(zscores)[i], np.diag(zscores_gaussian)[i]], 
             c='gray', alpha=0.5, lw=0.5)
plt.ylabel('$Z$-score', fontsize=15)
plt.yscale('log')
plt.yticks([0.1, 1, 10], ['$-10^{-1}$', '$-10^{0}$', '$-10^{1}$'], fontsize=15)
plt.ylim(0.05, 30)
plt.xlim(-1, 7.75)
plt.gca().invert_yaxis()
plt.show()


fig = plt.figure(figsize=(6,3))
plt.boxplot([np.diag(single_residuals)[:-1], np.diag(single_residuals_sparse)[:-1],
             np.diag(single_residuals)[:-1], np.diag(single_residuals_bernoulli)[:-1],
             np.diag(single_residuals)[:-1], np.diag(single_residuals_shuffled)[:-1],
             np.diag(single_residuals)[:-1], np.diag(single_residuals_gaussian)[:-1]], 
            positions=[0,0.75,2,2.75,4,4.75,6,6.75],
            showfliers=False, zorder=10)
plt.scatter(np.repeat(0, len(single_residuals)-1), np.diag(single_residuals)[:-1], 
            s=10, edgecolors='none', alpha=0.5, facecolors='gray')
plt.scatter(np.repeat(2, len(single_residuals)-1), np.diag(single_residuals)[:-1], 
            s=10, edgecolors='none', alpha=0.5, facecolors='gray')
plt.scatter(np.repeat(4, len(single_residuals)-1), np.diag(single_residuals)[:-1], 
            s=10, edgecolors='none', alpha=0.5, facecolors='gray')
plt.scatter(np.repeat(6, len(single_residuals)-1), np.diag(single_residuals)[:-1], 
            s=10, edgecolors='none', alpha=0.5, facecolors='gray')
plt.scatter(np.repeat(0.75, len(single_residuals)-1), np.diag(single_residuals_sparse)[:-1], 
            s=10, edgecolors='none', alpha=0.5, facecolors='gray')
plt.scatter(np.repeat(2.75, len(single_residuals)-1), np.diag(single_residuals_bernoulli)[:-1], 
            s=10, edgecolors='none', alpha=0.5, facecolors='gray')
plt.scatter(np.repeat(4.75, len(single_residuals)-1), np.diag(single_residuals_shuffled)[:-1], 
            s=10, edgecolors='none', alpha=0.5, facecolors='gray')
plt.scatter(np.repeat(6.75, len(single_residuals)-1), np.diag(single_residuals_gaussian)[:-1], 
            s=10, edgecolors='none', alpha=0.5, facecolors='gray')
plt.xticks([0,0.75,2,2.75,4,4.75,6,6.75], ['Original', 'Sparse', 'Original', 'Bernoulli',
                                           'Original', 'Shuffled', 'Original', 'Gaussian'], 
           fontsize=12, rotation=35, ha='right')
for i in range(len(single_residuals)-1):
    plt.plot([0, 0.75], [np.diag(single_residuals)[i], np.diag(single_residuals_sparse)[i]], 
             c='gray', alpha=0.5, lw=0.5)
    plt.plot([2, 2.75], [np.diag(single_residuals)[i], np.diag(single_residuals_bernoulli)[i]], 
             c='gray', alpha=0.5, lw=0.5)
    plt.plot([4, 4.75], [np.diag(single_residuals)[i], np.diag(single_residuals_shuffled)[i]], 
             c='gray', alpha=0.5, lw=0.5)
    plt.plot([6, 6.75], [np.diag(single_residuals)[i], np.diag(single_residuals_gaussian)[i]], 
             c='gray', alpha=0.5, lw=0.5)
plt.ylabel(r'self-residual ($r_{\alpha|\alpha}$)', fontsize=15)
plt.yscale('log')
plt.yticks([0.1, 1e-5, 1e-9], fontsize=15)
plt.ylim(1e-12, 3000)
plt.xlim(-1, 7.75)
plt.show()


print('wilcoxon')
print(scipy.stats.wilcoxon(np.diag(zscores)[:-1], np.diag(zscores_sparse)[:-1]))
print(scipy.stats.wilcoxon(np.diag(zscores)[:-1], np.diag(zscores_bernoulli)[:-1]))
print(scipy.stats.wilcoxon(np.diag(zscores)[:-1], np.diag(zscores_shuffled)[:-1]))
print(scipy.stats.wilcoxon(np.diag(zscores)[:-1], np.diag(zscores_gaussian)[:-1]))

print(scipy.stats.wilcoxon(np.diag(single_residuals)[:-1], np.diag(single_residuals_sparse)[:-1]))
print(scipy.stats.wilcoxon(np.diag(single_residuals)[:-1], np.diag(single_residuals_bernoulli)[:-1]))
print(scipy.stats.wilcoxon(np.diag(single_residuals)[:-1], np.diag(single_residuals_shuffled)[:-1]))
print(scipy.stats.wilcoxon(np.diag(single_residuals)[:-1], np.diag(single_residuals_gaussian)[:-1]))

print('mannwhitneyu')
print(scipy.stats.mannwhitneyu(np.diag(zscores)[:-1], np.diag(zscores_sparse)[:-1]))
print(scipy.stats.mannwhitneyu(np.diag(zscores)[:-1], np.diag(zscores_bernoulli)[:-1]))
print(scipy.stats.mannwhitneyu(np.diag(zscores)[:-1], np.diag(zscores_shuffled)[:-1]))
print(scipy.stats.mannwhitneyu(np.diag(zscores)[:-1], np.diag(zscores_gaussian)[:-1]))

print(scipy.stats.mannwhitneyu(np.diag(single_residuals)[:-1], np.diag(single_residuals_sparse)[:-1]))
print(scipy.stats.mannwhitneyu(np.diag(single_residuals)[:-1], np.diag(single_residuals_bernoulli)[:-1]))
print(scipy.stats.mannwhitneyu(np.diag(single_residuals)[:-1], np.diag(single_residuals_shuffled)[:-1]))
print(scipy.stats.mannwhitneyu(np.diag(single_residuals)[:-1], np.diag(single_residuals_gaussian)[:-1]))

