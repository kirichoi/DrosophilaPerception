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
# Figures related to uPN activity reconstruction in response to natural odor
# mixtures (fruits) and odorants at various concentrations are available in 
# this Python script.
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
import matplotlib
from collections import Counter
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

hallem_odor_sensitivity_conc_raw = pd.read_excel('./data/Hallem_and_Carlson_2006_TS2.xlsx')
hallem_odor_sensitivity_conc_raw = hallem_odor_sensitivity_conc_raw.fillna(0)

hallem_odor_sensitivity_conc = hallem_odor_sensitivity_conc_raw.iloc[:,1:]
hallem_odor_type_conc = hallem_odor_sensitivity_conc_raw['Odor']
hallem_odor_sensitivity_conc.index = hallem_odor_type_conc
hallem_PN_type_conc = hallem_odor_sensitivity_conc_raw.columns[1:]

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

master_odor_type = np.append(master_odor_type, hallem_odor_type_conc.iloc[10:40])
master_concentration = hallem_odor_type_conc.iloc[10:40].str[-2:]

natural_odor_type = hallem_odor_type_conc.iloc[40:]
natural_concentration = hallem_odor_type_conc.iloc[40:].str[-2:]

master_odor_sensitivity = np.zeros((len(master_odor_type)+1, len(master_PN_type)+1), dtype=object)
master_odor_sensitivity[:,0][1:] = master_odor_type
master_odor_sensitivity[0][1:] = master_PN_type

for i in range(len(hallem_odor_sensitivity_raw)):
    row = hallem_odor_sensitivity_raw.iloc[i]
    odor_idx = np.where(row[0]==master_odor_type)[0][0]+1
    for j in range(len(row)-1):
        PN_idx = np.where(hallem_PN_type[j]==master_PN_type)[0][0]+1
        master_odor_sensitivity[odor_idx][PN_idx] = row[j+1]
    
for i in np.arange(10,40):
    row = hallem_odor_sensitivity_conc_raw.iloc[i]
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

# Remove odors with no strong responses
for i,j in enumerate(master_odor_sensitivity_array):
    if len(np.where(np.abs(j) >= 40)[0]) != 0:
        good_idx.append(i)

master_odor_type = master_odor_type[good_idx]
master_odor_sensitivity_array = master_odor_sensitivity_array[good_idx]

master_odor_sensitivity_df = pd.DataFrame(master_odor_sensitivity_array)
master_odor_sensitivity_df.columns = master_PN_type
master_odor_sensitivity_df.index = master_odor_type

natural_odor_sensitivity = np.zeros((len(natural_odor_type)+1, len(hallem_PN_type)+1), dtype=object)
natural_odor_sensitivity[:,0][1:] = natural_odor_type
natural_odor_sensitivity[0][1:] = hallem_PN_type

for i in np.arange(36):
    row = hallem_odor_sensitivity_conc_raw.iloc[40+i]
    odor_idx = np.where(row[0]==natural_odor_type)[0][0]+1
    for j in range(len(row)-1):
        PN_idx = np.where(hallem_PN_type[j]==hallem_PN_type)[0][0]+1
        natural_odor_sensitivity[odor_idx][PN_idx] = row[j+1]    
    
natural_odor_sensitivity_array = natural_odor_sensitivity[1:,1:].astype(float)

DM35idx = np.where(hallem_PN_type == 'DM35')[0][0]
DM3idx = np.where(hallem_PN_type == 'DM3')[0][0]
DM5idx = np.where(hallem_PN_type == 'DM5')[0][0]

natural_odor_sensitivity_array[:,DM3idx] = natural_odor_sensitivity_array[:,DM3idx] + natural_odor_sensitivity_array[:,DM35idx]
natural_odor_sensitivity_array[:,DM5idx] = natural_odor_sensitivity_array[:,DM5idx] + natural_odor_sensitivity_array[:,DM35idx]

natural_odor_sensitivity_array = np.delete(natural_odor_sensitivity_array, DM35idx, axis=1)
natural_PN_type = np.delete(hallem_PN_type, DM35idx)

good_idx = []

# Remove odors with no strong responses
for i,j in enumerate(natural_odor_sensitivity_array):
    if len(np.where(np.abs(j) >= 40)[0]) != 0:
        good_idx.append(i)

natural_odor_type = np.array(natural_odor_type)[good_idx]
natural_odor_sensitivity_array = natural_odor_sensitivity_array[good_idx]

natural_odor_sensitivity_df = pd.DataFrame(natural_odor_sensitivity_array)
natural_odor_sensitivity_df.columns = natural_PN_type
natural_odor_sensitivity_df.index = natural_odor_type

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
    gidx = np.where(np.abs(spike) >= 40)[0]
    odor_glo.append(list(master_PN_type[gidx]))

Smat = np.zeros(np.shape(matrix_MBONKC.T))

for i in range(len(matrix_MBONKC.T)):
    Smat[i] = matrix_MBONKC.T[i]/np.linalg.norm(matrix_MBONKC.T[i])

Psimat = np.zeros(np.shape(matrix_KC_re.T))

for i in range(len(matrix_KC_re)):
    Psimat[:,i] = matrix_KC_re[i]/np.linalg.norm(matrix_KC_re[i])

Theta = np.matmul(Smat, Psimat)

odortruinput = []
odortruPN = []

for o in master_odor_type:
    spike = master_odor_sensitivity_df.loc[o].to_numpy()
    
    truPNactivity = np.zeros(len(KC_newidx_label))
    
    for i in range(len(alltarglo)):
        gloidx = np.where(KC_newidx_label == alltarglo[i])[0]
        if alltarglo[i] in master_PN_type:
            s = np.where(alltarglo[i] == master_PN_type)[0][0]
            truPNactivity[gloidx] = spike[s]
        
    odortruPN.append(truPNactivity)
    
    KCact = np.dot(Psimat, truPNactivity)
    
    y = np.dot(Smat, KCact)
    odortruinput.append(y)

master_odor_color = []

for o in master_odor_type[:-13]:
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

for o in master_odor_type[-13:]:
    if odor_chemtype_dict[o[:-3]] == 'amines':
        master_odor_color.append('#a01315')
    elif odor_chemtype_dict[o[:-3]] == 'lactones':
        master_odor_color.append('#f62124')
    elif odor_chemtype_dict[o[:-3]] == 'acids':
        master_odor_color.append('#fe4a91')
    elif odor_chemtype_dict[o[:-3]] == 'sulfur compounds':
        master_odor_color.append('#5602ac')
    elif odor_chemtype_dict[o[:-3]] == 'terpenes':
        master_odor_color.append('#ac5ffa')
    elif odor_chemtype_dict[o[:-3]] == 'aldehydes':
        master_odor_color.append('#153ac9')
    elif odor_chemtype_dict[o[:-3]] == 'ketones':
        master_odor_color.append('#0f9af0')
    elif odor_chemtype_dict[o[:-3]] == 'aromatics':
        master_odor_color.append('#0ff0ae')
    elif odor_chemtype_dict[o[:-3]] == 'alcohols':
        master_odor_color.append('#1fb254')
    elif odor_chemtype_dict[o[:-3]] == 'esters':
        master_odor_color.append('#60b82c')
    else:
        master_odor_color.append('#000000')

odor_sort_ind = [73, 97, 82, 99, 66, 102, 76, 98, 105, 45, 100, 106, 60, 101, 
                 107, 35, 103, 108, 40, 104, 109]

odor_sort_ind1 = [[73, 97], [82, 99], [66, 102], [76, 98, 105], [45, 100, 106], 
                  [60, 101, 107], [35, 103, 108], [40, 104, 109]]

labels = ['ethyl acetate -2', 'ethyl acetate -4', 'ethyl butyrate -2',
       'ethyl butyrate -4', '1-octen-3-ol -2', '1-octen-3-ol -4',
       'pentyl acetate -2', 'pentyl acetate -4', 'pentyl acetate -6',
       'methyl salicylate -2', 'methyl salicylate -4',
       'methyl salicylate -6', '1-hexanol -2', '1-hexanol -4',
       '1-hexanol -6', 'E2-hexenal -2', 'E2-hexenal -4', 'E2-hexenal -6',
       '2-heptanone -2', '2-heptanone -4', '2-heptanone -6']

labels2 = [['ethyl acetate -2', 'ethyl acetate -4'], ['ethyl butyrate -2',
       'ethyl butyrate -4'], ['1-octen-3-ol -2', '1-octen-3-ol -4'],
       ['pentyl acetate -2', 'pentyl acetate -4', 'pentyl acetate -6'],
       ['methyl salicylate -2', 'methyl salicylate -4',
       'methyl salicylate -6'], ['1-hexanol -2', '1-hexanol -4',
       '1-hexanol -6'], ['E2-hexenal -2', 'E2-hexenal -4', 'E2-hexenal -6'],
       ['2-heptanone -2', '2-heptanone -4', '2-heptanone -6']]

unu = []

for l in np.array(natural_odor_type)[9:18]:
    unu.append(l[:-3])


#%% Natural fruit odor mixtures

np.random.seed(1234)

natural_allsingleres = []
natural_singleinput = []
natural_result = []
natural_allsingletruPNactivity = []

for o in natural_odor_type:
    print(o)
    
    spike = copy.deepcopy(natural_odor_sensitivity_df.loc[o].to_numpy())
    spike[np.abs(spike) < 40] = 0
    
    truPNactivity = np.zeros(len(KC_newidx_label))
    
    for i in range(len(alltarglo)):
        gloidx = np.where(KC_newidx_label == alltarglo[i])[0]
        if alltarglo[i] in natural_PN_type:
            s = np.where(alltarglo[i] == natural_PN_type)[0][0]
            truPNactivity[gloidx] = spike[s]
        
    natural_allsingletruPNactivity.append(truPNactivity)
    
    KCact = np.dot(Psimat, truPNactivity)
    
    y = np.dot(Smat, KCact)
    natural_singleinput.append(y)
    
    bounds = scipy.optimize.Bounds(lb=-np.inf, ub=np.inf)
    constr = ({'type': 'eq', 'fun': lambda x: Theta @ x - y})
    
    x0 = np.linalg.pinv(Theta) @ y
    
    res = minimize(L1norm, x0, method='SLSQP', bounds=bounds, constraints=constr, options={'maxiter': 10000})
    
    natural_allsingleres.append(res.x)

conc_single_residuals = []
    
for l,i in enumerate(natural_allsingleres):
    rh_temp = []
    for k,j in enumerate(natural_singleinput):
        gidx = np.where(np.abs(natural_allsingletruPNactivity[k]) >= 40)[0]
        rh_temp.append(np.linalg.norm(j-np.dot(Smat,np.dot(Psimat[:,gidx], i[gidx])))/np.linalg.norm(j))
    conc_single_residuals.append(rh_temp)


#%% Figure 6A - Sparsity comparison between natural fruit odor mixtures and random mixtures

mp = 25

single_residuals = np.load('./precalc/single_residuals3.npy')

masked_array1 = copy.deepcopy(np.array(single_residuals))
for i,j in enumerate(masked_array1):
    idx_of_min = np.argmin(j)
    masked_array1[i][idx_of_min] = None

master_odor_success = master_odor_type[:-13][np.isnan(np.diag(masked_array1))][:-1]
master_odor_success = np.delete(master_odor_success, np.where(master_odor_success ==  'geosmin'))

hallemallsingletruPNactivity = []

for o in master_odor_success:
    spike = copy.deepcopy(hallem_odor_sensitivity.loc[o].to_numpy())
    spike[np.abs(spike) < 40] = 0

    truPNactivity = np.zeros(len(KC_newidx_label))

    for i in range(len(alltarglo)):
        gloidx = np.where(KC_newidx_label == alltarglo[i])[0]
        if alltarglo[i] in hallem_PN_type:
            s = np.where(alltarglo[i] == hallem_PN_type)[0][0]
            truPNactivity[gloidx] = spike[s]

    hallemallsingletruPNactivity.append(truPNactivity)

if not LOAD:
    np.random.seed(1234)
    
    hallem_allmulttruPNactivity = []
    
    for j,c in enumerate(np.arange(2, 15)):
        print(c)
        allres_temp = []
        alltruPNactivity_temp = []
        multcosine_temp = []
            
        randind = np.random.randint(len(master_odor_success), size=(500, c))
        
        for r in randind:
            odors = master_odor_success[r]
            spike = np.zeros(len(hallem_PN_type))
            
            for o in odors:
                s = copy.deepcopy(hallem_odor_sensitivity.loc[o].to_numpy())
                s[np.abs(s) < 40] = 0
                spike += s
            
            truPNactivity = np.zeros(len(KC_newidx_label))
    
            for i in range(len(alltarglo)):
                gloidx = np.where(KC_newidx_label == alltarglo[i])[0]
                if alltarglo[i] in hallem_PN_type:
                    s = np.where(alltarglo[i] == hallem_PN_type)[0][0]
                    truPNactivity[gloidx] = spike[s]
            
            alltruPNactivity_temp.append(truPNactivity)
            
        hallem_allmulttruPNactivity.append(alltruPNactivity_temp)
else:
    hallem_allmulttruPNactivity = np.load('./precalc/hallem_allmulttruPNactivity_500_3.npy')

natural_sparsity_hallem = list(Counter(np.nonzero(np.array(hallemallsingletruPNactivity))[0]).values())

natural_PNsparsity = []

for i in natural_allsingletruPNactivity:
    natural_PNsparsity.append(len(np.nonzero(i)[0]))
    
synthetic_PNsparsity = []

for i in hallem_allmulttruPNactivity:
    temp =[]
    for j in i:
        temp.append(len(np.nonzero(j)[0]))
    synthetic_PNsparsity.append(temp)

synthetic_PNsparsity_mean = []
synthetic_PNsparsity_std = []

for i in synthetic_PNsparsity:
    synthetic_PNsparsity_mean.append(np.mean(i))
    synthetic_PNsparsity_std.append(np.std(i))

cmap1 = matplotlib.cm.get_cmap('plasma')
cmap2 = matplotlib.cm.get_cmap('viridis')

fig, ax = plt.subplots(figsize=(2.5,3.5))
for i,j in enumerate(natural_PNsparsity[9:18]):
    b = np.random.random()
    if b>0.5:
        plt.hlines(j, 0, 15, color=cmap1(i/9), ls='--', zorder=i)
        plt.text(16, j-.015, unu[i], fontsize=12, color=cmap1(i/9))
    else:
        plt.hlines(j, 0, 15, color=cmap2(i/9), ls='--', zorder=i)
        plt.text(16, j-.015, unu[i], fontsize=12, color=cmap2(i/9))
plt.hlines(mp, 0, 15, color='k', ls='dotted')
plt.scatter(np.arange(2,15), synthetic_PNsparsity_mean, zorder=9, c='tab:red')
plt.errorbar(np.arange(2,15), synthetic_PNsparsity_mean, yerr=synthetic_PNsparsity_std, zorder=9, c='tab:red')
plt.scatter([1], np.mean(natural_sparsity_hallem), color='tab:red', marker='*', s=100, zorder=10)
plt.errorbar([1], np.mean(natural_sparsity_hallem), yerr=np.std(natural_sparsity_hallem), zorder=9, c='tab:red')
plt.xticks([0,5,10,15], fontsize=15)
plt.yticks([0,10,20,30,40], fontsize=15)
plt.ylabel('Sparsity $K$', fontsize=15)
plt.xlabel('$N_{od}$', fontsize=15)
plt.show()


#%% Figure 6D - Residuals for natural fruit odor mixtures 

custom_cmap = matplotlib.cm.get_cmap("RdYlBu").copy()
custom_cmap.set_bad(color='tab:red')

masked_array = copy.deepcopy(np.array(conc_single_residuals))

fig, ax = plt.subplots(figsize=(3,3))
im = plt.imshow(masked_array[9:18,9:18], cmap=custom_cmap, 
                norm=matplotlib.colors.LogNorm(vmax=1.1), interpolation='nearest')
ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_label_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_label_position('left')
plt.xticks(np.arange(9), unu, rotation='vertical', fontsize=10)
plt.yticks(np.arange(9), unu, fontsize=10)
cbar = plt.colorbar(im, fraction=0.04, location='top', pad=0.05)
cbar.ax.tick_params(labelsize=13)
cbar.set_ticks([1e0, 1e-4, 1e-8])
plt.tight_layout()
plt.show()


#%% Figure S4A - Z-scores for natural fruit odor mixtures

zscores = np.abs(scipy.stats.zscore(conc_single_residuals, axis=1))

fig, ax = plt.subplots(figsize=(3,2.5))
plt.bar(np.arange(len(unu)), np.diag(zscores)[9:18])
plt.xticks(np.arange(len(unu)), np.array(unu), rotation='vertical', fontsize=13)
plt.yscale('log')
plt.yticks(fontsize=15)
plt.ylabel('$Z$-score', fontsize=15)
plt.xlim(-1, len(unu))
plt.ylim(0.25, 11)
plt.yticks([1, 10], ['$-10^{0}$', '$-10^{1}$'], fontsize=15)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

#%% Figure S4B - Full MBON response profiles for natural mixtures

numpsp = 11

fig, ax = plt.subplots(1, numpsp, figsize=(5,4))
fig.delaxes(ax[9])
fig.delaxes(ax[10])
i1 = 0
i2 = 0
for i,j in enumerate(natural_singleinput[9:18]):
    if i2 == numpsp:
        i1 += 1
        i2 = 0
    ax[i2].set_title(natural_odor_type[9:18][i][:-3], rotation=90, fontsize=13)
    ax[i2].imshow(j[np.newaxis].T, cmap='binary', aspect='auto', interpolation='nearest', 
                      vmax=np.max(natural_singleinput[9:18]), vmin=np.min(natural_singleinput[9:18]))
    ax[i2].set_xticks([])
    ax[i2].set_yticks([])
    i2 += 1
plt.tight_layout()
plt.show()

#%% Figure 7A - Concentration uPN activity
                                 
for j,i in enumerate(odor_sort_ind1):
    a = 4.5/len(odor_sort_ind)*len(i)
    fig, ax = plt.subplots(figsize=(7,a))
    im = plt.imshow(master_odor_sensitivity_array[i],
                    vmin=-np.max(master_odor_sensitivity_array[odor_sort_ind]),
                    vmax=np.max(master_odor_sensitivity_array[odor_sort_ind]),
                    cmap='RdBu_r', aspect='equal')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_label_position('left')
    plt.yticks(np.arange(len(i)), labels2[j], fontsize=10)
    plt.xticks(np.arange(len(master_PN_type)), master_PN_type, fontsize=10, rotation='vertical')
    for ytick, color in zip(ax.get_yticklabels(), np.array(master_odor_color)[i]):
        ytick.set_color(color)
    cbar = plt.colorbar(im, fraction=0.3, location='top', pad=0.05)
    cbar.ax.tick_params(labelsize=15)
    plt.show()

#%% Odorants at different concentrations

np.random.seed(1234)

allsingleres = []
singleinput = []
allsingletruPNactivity = []

tt = np.arange(len(master_odor_type))
tt = np.delete(tt, odor_sort_ind)

for o in master_odor_type[tt]:
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


for o in master_odor_type[odor_sort_ind]:
    print(o)
    
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
    
    bounds = scipy.optimize.Bounds(lb=-np.inf, ub=np.inf)
    constr = ({'type': 'eq', 'fun': lambda x: Theta @ x - y})
    
    x0 = np.linalg.pinv(Theta) @ y
    
    res = minimize(L1norm, x0, method='SLSQP', bounds=bounds, constraints=constr, options={'maxiter': 10000})
    
    allsingleres.append(res.x)


conc_single_residuals = []
    
for l,i in enumerate(allsingleres):
    rh_temp = []
    for k,j in enumerate(singleinput):
        gidx = np.where(np.abs(allsingletruPNactivity[k]) >= 40)[0]
        rh_temp.append(np.linalg.norm(j-np.dot(Smat,np.dot(Psimat[:,gidx], i[gidx])))/np.linalg.norm(j))
    conc_single_residuals.append(rh_temp)


#%% Figure 7B - Concentration-dependent perception of odor

from matplotlib.gridspec import GridSpec

i2 = 0

labels = ['ethyl acetate', 'ethyl butyrate', '1-octen-3-ol', 'pentyl acetate', 
          'methyl salicylate', '1-hexanol', 'E2-hexenal', '2-heptanone']

concidx1 = np.arange(len(master_odor_type)-len(odor_sort_ind))
concidx2 = np.arange(len(master_odor_type)-len(odor_sort_ind), len(master_odor_type))

fig = plt.figure(figsize=(8,8))

gs = GridSpec(3, 8, figure=fig)
ax1 = fig.add_subplot(gs[0, :2])
ax2 = fig.add_subplot(gs[1, :2])
ax3 = fig.add_subplot(gs[2, :2])
ax4 = fig.add_subplot(gs[0, 2:5])
ax5 = fig.add_subplot(gs[0, 5:])
ax6 = fig.add_subplot(gs[1, 2:5])
ax7 = fig.add_subplot(gs[1, 5:])
ax8 = fig.add_subplot(gs[2, 2:5])

for i,j in enumerate(fig.get_axes()):
    if i < 3:
        rp = 2
    else:
        rp = 3
    for k in np.flip(np.arange(rp)):
        data = np.array(conc_single_residuals[i2])[np.append(concidx1, concidx2[i2])]
        box1 = j.boxplot(data, positions=[k],
                            widths=0.25,
                            patch_artist=True,
                            notch='',
                            showfliers=False,
                            boxprops={'fill': None}, zorder=3)
        x = np.random.normal(k, 0.1, size=len(data))
        j.scatter(x, data, marker='.', color=np.array(master_odor_color)[odor_sort_ind[i2]], edgecolors='none', alpha=0.5, s=80, zorder=2)
        j.scatter(x[-1], data[-1], marker='*', color='tab:red', s=100, zorder=9)
    
        for linetype in box1.values():
            for line in linetype:
                line.set_color('#616161')
                line.set_linewidth(1.5)
        
        i2 += 1
        
    if i < 3:
        j.set_xticks(np.arange(rp))
        j.set_xticklabels(['$10^{-4}$', '$10^{-2}$'], fontsize=15)
    else:
        j.set_xticks(np.arange(rp))
        j.set_xticklabels(['$10^{-6}$', '$10^{-4}$', '$10^{-2}$'], fontsize=15)

    j.set_ylim(-0.1, 1.5)
    if i > 2:
        j.set_yticks([])
    else:
        j.set_ylabel(r'$r_{\alpha|\beta}$', fontsize=25)
        j.set_yticks([0, 0.5, 1, 1.5])
        j.set_yticklabels([0, 0.5, 1, 1.5], fontsize=15)
    j.set_title(labels[i], 
                color=np.array(master_odor_color)[np.where(master_odor_type==labels[i])[0][0]],
                fontsize=15)

plt.tight_layout()
plt.show()


#%% Figure S5A - Concentration sparsity

conc_sparsity = np.empty((10,4))
iii = 0

for j,i in enumerate(hallem_odor_type_conc[:-36]):
    s = i.split(' ')
    if j < 10:
        i = i[:-3]
        idx = np.where(master_odor_type == i)[0][0]
        #c = len(np.nonzero(master_odor_sensitivity_array[idx])[0])
        tar = np.abs(master_odor_sensitivity_array[idx]) > 40
        c = 0
        for k in master_PN_type[tar]:
            c += len(np.where(KC_newidx_label == k)[0])
    else:
        idx = np.where(master_odor_type == i)[0]
        if len(idx) > 0:
            # c = len(np.nonzero(master_odor_sensitivity_array[idx[0]])[0])
            tar = np.abs(master_odor_sensitivity_array[idx[0]]) > 40
            c = 0
            for k in master_PN_type[tar]:
                c += len(np.where(KC_newidx_label == k)[0])
        else:
            c = 0
    if iii >= 10:
        iii = 0
    conc_sparsity[iii][int((-1*int(s[-1])-2)/2)] = c
    iii += 1

sid = [9,7,0,3,6,2,5,8,1,4]
sid = [4, 1, 8, 5, 2, 6, 3, 0, 7, 9]

conc_sparsity = conc_sparsity[sid]

nun = []

for l in hallem_odor_type_conc[:10]:
    nun.append(l[:-3])

nun = np.array(nun)[sid]

_xx, _yy = np.meshgrid(np.flip(np.arange(4))-.3, np.arange(10)-0.75)
x, y = _xx.ravel(), _yy.ravel()

def polygon_under_graph(x, y):
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]

ax = plt.figure(figsize=(7,7)).add_subplot(projection='3d')
lambdas = range(1, 9)
verts = [polygon_under_graph(x, l) for l in conc_sparsity]
facecolors = plt.colormaps['plasma_r'](np.linspace(0.1, 0.9, len(conc_sparsity)))
facecolors = facecolors[np.random.choice(np.arange(10), 10, replace=False)]
ax.bar3d(np.flip(y), x, np.zeros(len(conc_sparsity.flatten())), 1, 0.5, conc_sparsity.flatten(), 
         color=np.repeat(facecolors, 4, axis=0), alpha=0.7, shade=False)
ax.set(ylim=(-0.5, 3.5), xlim=(-.5, 9.), zlim=(-1, 40))
ax.set_yticks(np.arange(4))
ax.set_yticklabels(np.flip(['$10^{-2}$','$10^{-4}$','$10^{-6}$','$10^{-8}$']), fontsize=13)
ax.set_xticks(np.arange(10))
ax.set_xticklabels(np.flip(nun), fontsize=12, ha='right', va='baseline')
for xtick, color in zip(ax.get_xticklabels(), np.flip(facecolors,axis=0)):
    xtick.set_color(color)
ax.set_zticks([0,10,20,30,40])
ax.set_zticklabels([0,10,20,30,40], fontsize=13)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.set_zlabel('Sparsity $K$', fontsize=15)
ax.set_ylabel('Dilution', fontsize=15, labelpad=10)
ax.set_box_aspect((5,5,3))
plt.tight_layout()
ax.view_init(40, -40)
ax.set_box_aspect(None, zoom=0.8)
plt.show()


#%% Figure S5B - Residuals for undiluted natural fruit odor mixtures

fig, ax = plt.subplots(figsize=(3,3))
im = plt.imshow(masked_array[:9,:9], cmap=custom_cmap, norm=matplotlib.colors.LogNorm(vmax=1.1, vmin=1e-8))
ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_label_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_label_position('left')
plt.xticks(np.arange(9), unu, rotation='vertical', fontsize=10)
plt.yticks(np.arange(9), unu, fontsize=10)
cbar = plt.colorbar(im, fraction=0.04, location='top', pad=0.05)
cbar.ax.tick_params(labelsize=13)
cbar.set_ticks([1e0, 1e-4, 1e-8])
plt.tight_layout()
plt.show()


#%% Figure S5C - Full MBON response profiles for odorants at different concentrations

numpsp = 11

fig, ax = plt.subplots(2, numpsp, figsize=(6,10))
fig.delaxes(ax[1][10])
i1 = 0
i2 = 0
for i,j in enumerate(singleinput[-len(odor_sort_ind):]):
    if i2 == numpsp:
        i1 += 1
        i2 = 0
    if '-' not in master_odor_type[odor_sort_ind][i]:
        lab = '{} -2'.format(master_odor_type[odor_sort_ind][i])
    else:
        lab = master_odor_type[odor_sort_ind][i]
    ax[i1][i2].set_title(lab, rotation=90, fontsize=13)
    ax[i1][i2].imshow(j[np.newaxis].T, cmap='binary', aspect='auto', interpolation='nearest', 
                      vmax=np.max(singleinput[-len(odor_sort_ind):]), 
                      vmin=np.min(singleinput[-len(odor_sort_ind):]))
    ax[i1][i2].set_xticks([])
    ax[i1][i2].set_yticks([])
    i2 += 1
plt.tight_layout()
plt.show()




