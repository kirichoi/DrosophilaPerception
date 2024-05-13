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
# Figures related to uPN activity reconstruction in response to single odorants
# and random odorant mixtures are available in this Python script.
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
from itertools import combinations
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

#%% Figure 3 - uPN activity profile of single odorants

fig, ax = plt.subplots(figsize=(7,14))
im = plt.imshow(master_odor_sensitivity_array, norm=matplotlib.colors.CenteredNorm(), cmap='RdBu_r', aspect='auto')
ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_label_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_label_position('left')
plt.xticks(np.arange(len(master_PN_type)), master_PN_type, rotation='vertical', fontsize=10)
plt.yticks(np.arange(len(master_odor_type)), master_odor_type, fontsize=10)
for ytick, color in zip(ax.get_yticklabels(), master_odor_color):
    ytick.set_color(color)
cbar = plt.colorbar(im, fraction=0.04, location='top', pad=0.01)
cbar.ax.tick_params(labelsize=15)
plt.tight_layout()
plt.show()

#%% Residual calculation for single odorants

if not LOAD:
    np.random.seed(1234)
    
    allsingleres = []
    
    for i,o in enumerate(master_odor_type):
        print(o)
        
        truPNactivity = allsingletruPNactivity[i]
        
        y = singleinput[i]
        
        bounds = scipy.optimize.Bounds(lb=-np.inf, ub=np.inf)
        constr = ({'type': 'eq', 'fun': lambda x: Theta @ x - y})
        
        x0 = np.linalg.pinv(Theta) @ y
        
        res = minimize(L1norm, x0, method='SLSQP', bounds=bounds, constraints=constr, options={'maxiter': 10000})
        
        allsingleres.append(res.x)
        
    single_residuals = []
        
    for l,i in enumerate(allsingleres):
        r_temp = []
        for k,j in enumerate(singleinput):
            gidx = np.where(np.abs(allsingletruPNactivity[k]) >= 40)[0]
            r_temp.append(np.linalg.norm(j-np.dot(Smat,np.dot(Psimat[:,gidx], i[gidx])))/np.linalg.norm(j))
        single_residuals.append(r_temp)
else:
    single_residuals = np.load('./precalc/single_residuals3.npy')

masked_array = copy.deepcopy(np.array(single_residuals))
masked_array1 = copy.deepcopy(np.array(single_residuals))
for i,j in enumerate(masked_array1):
    idx_of_min = np.argmin(j)
    masked_array1[i][idx_of_min] = None

natural_sparsity = list(Counter(np.nonzero(np.array(allsingletruPNactivity))[0]).values())

x = np.arange(len(master_odor_type))
cidx = np.where(np.isnan(np.diag(masked_array1)))[0]
ncidx = np.delete(x,cidx)


#%% Figure 4 - Full residual distributions

numpsp = 20
fig, ax = plt.subplots(5, 1, figsize=(20,20))
i1 = 0
i2 = 0
for i,j in enumerate(single_residuals):
    if i1 == numpsp:
        i1 = 0
        i2 += 1
    j = np.array(j)
    box1 = ax[i2].boxplot(j, positions=[i1], 
                       widths=0.25,
                       patch_artist=True,
                       notch='',
                       showfliers=False,
                       boxprops={'fill': None}, zorder=3)
    x = np.random.normal(i1, 0.1, size=len(j))
    ax[i2].scatter(x, j, marker='.', color=master_odor_color[i], edgecolors='none', alpha=0.5, s=80, zorder=2)
    ax[i2].scatter(x[i], j[i], marker='*', color='tab:red', s=100, zorder=9)

    for linetype in box1.values():
        for line in linetype:
            line.set_color('#616161')
            line.set_linewidth(1.5)
    i1 += 1 
        
for i in range(5):
    ax[i].set_ylim(-0.1, 1.5)
    ax[i].set_ylabel(r'$r_{\alpha|\beta}$', fontsize=25)
    if i == 4:
        ax[i].set_xticks(np.arange(17))
    else:
        ax[i].set_xticks(np.arange(numpsp))
    ax[i].set_xticklabels(master_odor_type[i*numpsp:(i+1)*numpsp], rotation=30, ha='right', fontsize=15)
    ax[i].set_yticks([0, 0.5, 1, 1.5])
    ax[i].set_yticklabels([0, 0.5, 1, 1.5], fontsize=20)
    for xtick, color in zip(ax[i].get_xticklabels(), master_odor_color[i*numpsp:(i+1)*numpsp]):
        xtick.set_color(color)
plt.tight_layout()
plt.show()

#%% Figure 5A - Z-scores for single odorants

zscores = -scipy.stats.zscore(single_residuals, axis=1)

fig, ax = plt.subplots(figsize=(3,16))
plt.barh(np.arange(len(single_residuals)), np.diag(zscores), color=master_odor_color)
ax.yaxis.set_ticks_position('right')
ax.yaxis.set_label_position('right')
plt.yticks(np.arange(len(master_odor_type)), np.array(master_odor_type), fontsize=10)
for ytick, color in zip(ax.get_yticklabels(), master_odor_color):
    ytick.set_color(color)
plt.xscale('log')
plt.xticks([1, 10], ['$-10^{0}$', '$-10^{1}$'], fontsize=15)
plt.xlabel('$Z$-score', fontsize=15)
plt.ylim(-1, len(master_odor_type))
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()

#%% Figure 5B - Sparsity vs self-residual

def sigmoid(x, L, k, x0, b):
    return L/(1 + np.exp(-k*(x-x0))) + b

popt, pcov = scipy.optimize.curve_fit(sigmoid, natural_sparsity, np.log10(np.diag(single_residuals)),
                                      p0=[-10,1,25,-1])

x = np.arange(np.max(natural_sparsity))
hp = (np.max(sigmoid(x, *popt)) - np.min(sigmoid(x, *popt)))/2 + np.min(sigmoid(x, *popt))
mp = x[np.argmin(np.abs(hp - sigmoid(x, *popt)))]

fig, ax = plt.subplots(figsize=(3.5,3))
plt.vlines(mp, 1e-10, 1, color='k', ls='--')
plt.scatter(np.array(natural_sparsity)[cidx], np.diag(single_residuals)[cidx], 
            facecolors='none', edgecolors=np.array(master_odor_color)[cidx], marker='o')
plt.scatter(np.array(natural_sparsity)[ncidx], np.diag(single_residuals)[ncidx],
            facecolors='none', edgecolors=np.array(master_odor_color)[ncidx], marker='*', s=60)
plt.xticks(fontsize=15)
plt.xlabel('Sparsity $K$', fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(1.5e-10, 0.9)
plt.yscale('log')
plt.ylabel('$r_{\\alpha|\\alpha}$', fontsize=20)
plt.tight_layout()
plt.show()

#%% Figure 5C - Sparsity vs Z-scores

p_r_1 = scipy.stats.pearsonr(natural_sparsity, np.log10(np.diag(zscores)))

fig, ax = plt.subplots(figsize=(3.5,3))
plt.scatter(np.array(natural_sparsity)[cidx], np.diag(zscores)[cidx], 
            facecolors='none', edgecolors=np.array(master_odor_color)[cidx], marker='o')
plt.scatter(np.array(natural_sparsity)[ncidx], np.diag(zscores)[ncidx], 
            facecolors='none', edgecolors=np.array(master_odor_color)[ncidx], marker='*', s=60)
plt.xticks(fontsize=15)
plt.xlabel('Sparsity $K$', fontsize=15)
plt.yticks(fontsize=15)
plt.yscale('log')
plt.ylabel('$Z$-score', fontsize=15)
plt.yticks([1, 10], ['$-10^{0}$', '$-10^{1}$'], fontsize=15)
plt.gca().invert_yaxis()
plt.text(20, 7, '$r = '+str(-np.around(p_r_1[0], 3))+'$', fontsize=15)
plt.text(20, 10, '$p \ll 0.0001$', fontsize=15)
plt.tight_layout()
plt.show()

#%% Figure 5D - Functional group sparsity comparison

alcohol_idx1 = np.where(master_odor_type == 'methanol')[0][0]
alcohol_idx2 = np.where(master_odor_type == '2,3-butanediol')[0][0]

alcohol_sensitivity = np.array(allsingletruPNactivity)[alcohol_idx1:alcohol_idx2+1]

ester_idx1 = np.where(master_odor_type == 'methyl acetate')[0][0]
ester_idx2 = np.where(master_odor_type == 'diethyl succinate')[0][0]

ester_sensitivity = np.array(allsingletruPNactivity)[ester_idx1:ester_idx2+1]

AT_idx1 = np.where(master_odor_type == 'acetic acid')[0][0]
AT_idx2 = np.where(master_odor_type == 'linalool oxide')[0][0]
AT_idx3 = np.where(master_odor_type == '3-methylthio-1-propanol')[0][0]
AT_idx4 = np.where(master_odor_type == 'dimethyl sulfide')[0][0]
AT_idx5 = np.arange(AT_idx1, AT_idx2+1)
AT_idx5 = np.setdiff1d(AT_idx5, [AT_idx3, AT_idx4])

AT_sensitivity = np.array(allsingletruPNactivity)[AT_idx5]

alcohol_Counter = Counter(np.where(alcohol_sensitivity >= 40)[0])
ester_Counter = Counter(np.where(ester_sensitivity >= 40)[0])
AT_Counter = Counter(np.where(AT_sensitivity >= 40)[0])


fig, ax = plt.subplots(figsize=(1.5,3))
plt.bar(np.arange(3), 
        [np.mean(list(alcohol_Counter.values())), 
        np.mean(list(ester_Counter.values())), 
        np.mean(list(AT_Counter.values()))],
        0.75,
        yerr=[np.std(list(alcohol_Counter.values())), 
            np.std(list(ester_Counter.values())), 
            np.std(list(AT_Counter.values()))],
        capsize=5)
plt.yticks([0, 10, 20, 30, 40], fontsize=13)
plt.xticks(np.arange(3), ['Alcohols', 'Esters', 'Acids+\nTerpenes'], rotation=45, horizontalalignment='right', fontsize=13)
plt.ylabel('Average sparsity $\\langle K \\rangle$', fontsize=15)
plt.show()

#%% Figure 5E - Methanol and ethanol residuals

fig, ax = plt.subplots(figsize=(3,3.5))
box1 = plt.boxplot(single_residuals[55], positions=[0], 
                   widths=0.25,
                   patch_artist=True,
                   notch='',
                   showfliers=False,
                   boxprops={'fill': None})
x = np.random.normal(0, 0.04, size=len(single_residuals))
plt.scatter(x, single_residuals[55], marker='.', color='tab:blue', alpha=0.5, s=100, edgecolors='none')
plt.scatter(x[55], single_residuals[55][55], marker='.', color='tab:red', s=60)
plt.scatter(x[56], single_residuals[55][56], marker='.', color='tab:orange', s=60)
box2 = plt.boxplot(single_residuals[56], positions=[0.5], 
                   widths=0.25,
                   patch_artist=True,
                   notch='',
                   showfliers=False,
                   boxprops={'fill': None})
x = np.random.normal(0.5, 0.04, size=len(single_residuals))
plt.scatter(x, single_residuals[56], marker='.', color='tab:blue', alpha=0.4, s=100, edgecolors='none')
plt.scatter(x[56], single_residuals[56][56], marker='.', color='tab:green', s=60)
plt.scatter(x[55], single_residuals[56][55], marker='.', color='tab:purple', s=60)

for linetype in box1.values():
    for line in linetype:
        line.set_color('#616161')
        line.set_linewidth(1.5)

for linetype in box2.values():
    for line in linetype:
        line.set_color('#616161')
        line.set_linewidth(1.5)

plt.ylim(-0.1, 1.1)
plt.xlim(-0.2, 0.7)
plt.ylabel('Residuals', fontsize=15)
plt.xticks([0, 0.5], ['Methanol', 'Ethanol'], fontsize=13)
plt.yticks([0, 0.5, 1], fontsize=13)
plt.show()


#%% Figure S3 - Residuals for single odorants in matrix form

custom_cmap = matplotlib.cm.get_cmap("RdYlBu").copy()
custom_cmap.set_bad(color='tab:red')

fig, ax = plt.subplots(figsize=(16,16))
im = plt.imshow(masked_array, cmap=custom_cmap, norm=matplotlib.colors.LogNorm(vmax=1.1), interpolation='nearest')
ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_label_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_label_position('left')
plt.xticks(np.arange(len(master_odor_type)), np.array(master_odor_type), rotation='vertical', fontsize=10)
plt.yticks(np.arange(len(master_odor_type)), np.array(master_odor_type), fontsize=10)
for xtick, color in zip(ax.get_xticklabels(), master_odor_color):
    xtick.set_color(color)
for ytick, color in zip(ax.get_yticklabels(), master_odor_color):
    ytick.set_color(color)
cbar = plt.colorbar(im, fraction=0.04, location='top', pad=0.01)
cbar.ax.tick_params(labelsize=15)
plt.show()


#%% Random odor mixture

if not LOAD:
    np.random.seed(1234)
    
    allmultres = []
    allmulttruPNactivity = []
    multcosine = []
    
    master_odor_success = master_odor_type[np.isnan(np.diag(masked_array1))][:-1]
    
    for j,c in enumerate(np.arange(2, 15)):
        print(c)
        allres_temp = []
        alltruPNactivity_temp = []
        multcosine_temp = []
            
        randind = np.random.randint(len(master_odor_success), size=(500, c))
        
        for r in randind:
            odors = master_odor_success[r]
            spike = np.zeros(len(master_PN_type))
            
            for o in odors:
                s = copy.deepcopy(master_odor_sensitivity_df.loc[o].to_numpy())
                s[np.abs(s) < 40] = 0
                spike += s
            
            truPNactivity = np.zeros(len(KC_newidx_label))
    
            for i in range(len(alltarglo)):
                gloidx = np.where(KC_newidx_label == alltarglo[i])[0]
                if alltarglo[i] in master_PN_type:
                    s = np.where(alltarglo[i] == master_PN_type)[0][0]
                    truPNactivity[gloidx] = spike[s]
            
            alltruPNactivity_temp.append(truPNactivity)
            
            KCact = np.dot(Psimat, truPNactivity)
            
            y = np.dot(Smat, KCact)
            
            bounds = scipy.optimize.Bounds(lb=-np.inf, ub=np.inf)
            constr = ({'type': 'eq', 'fun': lambda x: Theta @ x - y})
            
            x0 = np.linalg.pinv(Theta) @ y
            
            res = minimize(L1norm, x0, method='SLSQP', bounds=bounds, constraints=constr, options={'maxiter': 10000})
            
            allres_temp.append(res.x)
            multcosine_temp.append(scipy.spatial.distance.cosine(truPNactivity, res.x))
            
        allmultres.append(allres_temp)
        allmulttruPNactivity.append(alltruPNactivity_temp)
        multcosine.append(multcosine_temp)

    mult_cosine_error = []
    
    for l,i in enumerate(allmultres):
        result_temp = []
        for x,y, in enumerate(i):
            result_temp.append(multcosine[l][x]<=0.05)
        mult_cosine_error.append(result_temp)
else:
    mult_cosine_error = np.load(r'./precalc/mult_cosine_error_500_3.npy')
    allmulttruPNactivity = np.load(r'./precalc/allmulttruPNactivity_500_3.npy')

trues = [1]
for i in mult_cosine_error:
    trues.append(Counter(i)[True]/500)

synthetic_PNsparsity = []

for i in allmulttruPNactivity:
    temp =[]
    for j in i:
        temp.append(len(np.nonzero(j)[0]))
    synthetic_PNsparsity.append(temp)

synthetic_PNsparsity_mean = []
synthetic_PNsparsity_std = []

for i in synthetic_PNsparsity:
    synthetic_PNsparsity_mean.append(np.mean(i))
    synthetic_PNsparsity_std.append(np.std(i))    

#%% Figure 6B - Random sampling of naturalistic odorants

fig, ax = plt.subplots(figsize=(3.75,3))
ax2 = ax.twinx()
ax.plot(np.arange(1, 15), trues, lw=3)
ax2.hlines(mp, 0, 15, color='k', ls='--')
ax2.scatter(np.arange(2,15), synthetic_PNsparsity_mean, zorder=9, c='tab:red')
ax2.errorbar(np.arange(2,15), synthetic_PNsparsity_mean, yerr=synthetic_PNsparsity_std, zorder=9, c='tab:red')
ax2.scatter([1], np.mean(natural_sparsity), color='tab:red', marker='*', s=100, zorder=10)
ax2.errorbar([1], np.mean(natural_sparsity), yerr=np.std(natural_sparsity), zorder=10, c='tab:red')
ax.set_ylabel('% correct', fontsize=15)
ax.set_xlabel('$N_{od}$', fontsize=15)
ax.set_xticks([0,5,10,15])
ax.set_xticklabels([0,5,10,15], fontsize=15)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
ax.set_xlim(0, 15)
ax.yaxis.label.set_color('tab:blue')
ax.tick_params(axis='y', colors='tab:blue')
ax2.set_yticks([0, 10, 20, 30, 40, 50])
ax2.set_yticklabels([0, 10, 20, 30, 40, 50], fontsize=15)
ax2.set_ylabel('Sparsity $K$', fontsize=15)
ax2.yaxis.label.set_color('tab:red')
ax2.tick_params(axis='y', colors='tab:red')
plt.tight_layout()
plt.show()

#%% Combinatorial odorant mixture based on odor valence 

# good smell
odor_attr = ['g-butyrolactone', '2,3-butanedione', 'propionic acid', 'phenylacetaldehyde',
             '1-pentanol', 'methyl acetate', 'propyl acetate']

odor_attr_p = ['ethyl acetate', '2-heptanone']

odor_attr = odor_attr + odor_attr_p

# bad smell
odor_avers = ['1-octen-3-ol', '1-octanol', 'linalool', 'benzaldehyde', 'geosmin',
              'methyl salicylate']

odor_avers_e = ['2-methylphenol']

odor_avers = odor_avers + odor_avers_e

if not LOAD:
    attr_comb = [list(x) for x in combinations(np.array(odor_attr), 2)]
    avers_comb = [list(x) for x in combinations(np.array(odor_avers), 2)]
    
    oppose_comb = []
     
    for i in range(len(odor_attr)):
        for j in range(len(odor_avers)):
            oppose_comb.append([odor_attr[i], odor_avers[j]])
    
    np.random.seed(1234)
    
    multcosine_attr = []
    allmultres_attr = []
    
    for odors in attr_comb:
        spike = np.zeros(len(master_PN_type))
        
        for o in odors:
            s = copy.deepcopy(master_odor_sensitivity_df.loc[o].to_numpy())
            s[np.abs(s) < 40] = 0
            spike += s
        
        truPNactivity = np.zeros(len(KC_newidx_label))
    
        for i in range(len(alltarglo)):
            gloidx = np.where(KC_newidx_label == alltarglo[i])[0]
            if alltarglo[i] in master_PN_type:
                s = np.where(alltarglo[i] == master_PN_type)[0][0]
                truPNactivity[gloidx] = spike[s]
        
        KCact = np.dot(Psimat, truPNactivity)
        
        y = np.dot(Smat, KCact)
        
        bounds = scipy.optimize.Bounds(lb=-np.inf, ub=np.inf)
        constr = ({'type': 'eq', 'fun': lambda x: Theta @ x - y})
        
        x0 = np.linalg.pinv(Theta) @ y
        
        res = minimize(L1norm, x0, method='SLSQP', bounds=bounds, constraints=constr, options={'maxiter': 10000})
        
        allmultres_attr.append(res.x)
        multcosine_attr.append(scipy.spatial.distance.cosine(truPNactivity, res.x))
    
    mult_cosine_error_attr = []
    
    for l,i in enumerate(allmultres_attr):
        mult_cosine_error_attr.append(multcosine_attr[l]<=0.05)
    
    
    allmultres_avers = []
    multcosine_avers = []
    
    for odors in avers_comb:
        spike = np.zeros(len(master_PN_type))
        
        for o in odors:
            s = copy.deepcopy(master_odor_sensitivity_df.loc[o].to_numpy())
            s[np.abs(s) < 40] = 0
            spike += s
        
        truPNactivity = np.zeros(len(KC_newidx_label))
    
        for i in range(len(alltarglo)):
            gloidx = np.where(KC_newidx_label == alltarglo[i])[0]
            if alltarglo[i] in master_PN_type:
                s = np.where(alltarglo[i] == master_PN_type)[0][0]
                truPNactivity[gloidx] = spike[s]
        
        KCact = np.dot(Psimat, truPNactivity)
        
        y = np.dot(Smat, KCact)
        
        bounds = scipy.optimize.Bounds(lb=-np.inf, ub=np.inf)
        constr = ({'type': 'eq', 'fun': lambda x: Theta @ x - y})
        
        x0 = np.linalg.pinv(Theta) @ y
        
        res = minimize(L1norm, x0, method='SLSQP', bounds=bounds, constraints=constr, options={'maxiter': 10000})
    
        allmultres_avers.append(res.x)
        multcosine_avers.append(scipy.spatial.distance.cosine(truPNactivity, res.x))
    
    mult_cosine_error_avers = []
    
    for l,i in enumerate(allmultres_avers):
        mult_cosine_error_avers.append(multcosine_avers[l]<=0.05)
else:
    mult_cosine_error_attr = np.load('./precalc/mult_cosine_error_attr3.npy')
    mult_cosine_error_avers = np.load('./precalc/mult_cosine_error_avers3.npy')

#%% Figure 6C - Comparison between attractive and aversive odorant mixtures

fig, ax = plt.subplots(figsize=(2.5,1))
a = [Counter(mult_cosine_error_attr)[True]/len(mult_cosine_error_attr),
     Counter(mult_cosine_error_avers)[True]/len(mult_cosine_error_avers)]
plt.barh(np.arange(2), a, color=['tab:green', 'tab:red'])
plt.xlabel('% correct', fontsize=15)
plt.yticks([0,1], ['Attractive', 'Aversive'], fontsize=15)
plt.xticks([0.4, 0.5, 0.6, 0.7], fontsize=15)
plt.ylim(-0.5, 1.5)
plt.xlim(0.4, 0.7)
for ytick, color in zip(ax.get_yticklabels(), ['tab:green', 'tab:red']):
    ytick.set_color(color)
plt.show()


#%% Figure S2 - Full MBON response profiles

numpsp = 33

fig, ax = plt.subplots(3, numpsp, figsize=(20,23))
fig.delaxes(ax[2][31])

fig.delaxes(ax[2][32])
i1 = 0
i2 = 0
for i,j in enumerate(singleinput):
    if i2 == numpsp:
        i1 += 1
        i2 = 0
    ax[i1][i2].set_title(master_odor_type[i], rotation=90, fontsize=20, 
                         color=master_odor_color[i], pad=10)
    ax[i1][i2].imshow(j[np.newaxis].T, cmap='binary', aspect='auto', 
                      interpolation='nearest', vmax=np.max(singleinput), 
                      vmin=np.min(singleinput))
    ax[i1][i2].set_xticks([])
    ax[i1][i2].set_yticks([])
    i2 += 1
fig.tight_layout()
plt.show()

#%% Figure 10A - Comparison between uPN and MBON response using Euclidean distance

from sklearn import metrics

MBONdist1 = scipy.spatial.distance.cdist(singleinput, singleinput)
MBONdis2 = metrics.pairwise.cosine_distances(singleinput)

PNdist1 = scipy.spatial.distance.cdist(allsingletruPNactivity, allsingletruPNactivity)
PNdist2 = metrics.pairwise.cosine_distances(allsingletruPNactivity)

testodor = ['geosmin', 'acetic acid', 'benzaldehyde', 'propyl acetate', '1-pentanol']
testodoridx = []

for i in testodor:
    testodoridx.append(np.where(master_odor_type == i)[0][0])

fig, ax = plt.subplots(figsize=(5,3))
i1 = 0
i2 = 0
for i,j in enumerate(MBONdist1[testodoridx]):
    j = np.array(j)/np.max(j)
    box1 = ax.boxplot(j, positions=[i1], 
                       widths=0.25,
                       patch_artist=True,
                       notch='',
                       showfliers=False,
                       boxprops={'fill': None}, zorder=3)
    x = np.random.normal(i1, 0.1, size=len(j))
    ax.scatter(x, j, marker='.', color=np.array(master_odor_color)[testodoridx][i], edgecolors='none', alpha=0.5, s=80, zorder=2)
    ax.scatter(x[testodoridx[i]], j[testodoridx[i]], marker='*', color='tab:red', s=100, zorder=9)

    for linetype in box1.values():
        for line in linetype:
            line.set_color('#616161')
            line.set_linewidth(1.5)
    i1 += 1 

ax.set_ylim(-0.1, 1.2)
ax.set_ylabel(r'$\hat{d}$', fontsize=15)
ax.set_xticklabels(testodor, rotation=30, ha='right', fontsize=13)
ax.set_yticks([0, 0.5, 1])
ax.set_yticklabels([0, 0.5, 1], fontsize=13)
for xtick, color in zip(ax.get_xticklabels(), np.array(master_odor_color)[testodoridx]):
    xtick.set_color(color)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(5,3))
i1 = 0
i2 = 0
for i,j in enumerate(PNdist1[testodoridx]):
    j = np.array(j)/np.max(j)
    box1 = ax.boxplot(j, positions=[i1], 
                       widths=0.25,
                       patch_artist=True,
                       notch='',
                       showfliers=False,
                       boxprops={'fill': None}, zorder=3)
    x = np.random.normal(i1, 0.1, size=len(j))
    ax.scatter(x, j, marker='.', color=np.array(master_odor_color)[testodoridx][i], edgecolors='none', alpha=0.5, s=80, zorder=2)
    ax.scatter(x[testodoridx[i]], j[testodoridx[i]], marker='*', color='tab:red', s=100, zorder=9)

    for linetype in box1.values():
        for line in linetype:
            line.set_color('#616161')
            line.set_linewidth(1.5)
    i1 += 1 

ax.set_ylim(-0.1, 1.2)
ax.set_ylabel(r'$\hat{d}$', fontsize=15)
ax.set_xticklabels(testodor, rotation=30, ha='right', fontsize=13)
ax.set_yticks([0, 0.5, 1])
ax.set_yticklabels([0, 0.5, 1], fontsize=13)
for xtick, color in zip(ax.get_xticklabels(), np.array(master_odor_color)[testodoridx]):
    xtick.set_color(color)
plt.tight_layout()
plt.show()

#%% Reconstruction of uPN activity when Gaussian noise is added

if not LOAD:
    np.random.seed(1234)
    
    alltarglo = np.unique(KC_newidx_label)
    
    noiselevel_result = [1]
    
    for noiselevel in np.arange(0.5,3.5,0.5):
        print(noiselevel)
        
        allsingletruPNactivity_t = []
        allsingleres_t = []
    
        for o in master_odor_type:
            spike = copy.deepcopy(master_odor_sensitivity_df.loc[o].to_numpy())
            spike[np.abs(spike) < 40] = 0
            
            truPNactivity = np.zeros(len(KC_newidx_label))
            
            for i in range(len(alltarglo)):
                gloidx = np.where(KC_newidx_label == alltarglo[i])[0]
                if alltarglo[i] in master_PN_type:
                    s = np.where(alltarglo[i] == master_PN_type)[0][0]
                    truPNactivity[gloidx] = spike[s]
                else:
                    truPNactivity[gloidx] = np.random.normal(0, noiselevel, len(gloidx))
                
            allsingletruPNactivity_t.append(truPNactivity)
        
        for oi,o in enumerate(master_odor_type[np.where(np.isnan(np.diag(masked_array1)))][:-1]):
            KCact = np.dot(Psimat, allsingletruPNactivity_t[oi])
            
            y = np.dot(Smat, KCact)
            
            bounds = scipy.optimize.Bounds(lb=-np.inf, ub=np.inf)
            constr = ({'type': 'eq', 'fun': lambda x: Theta @ x - y})
            
            x0 = np.linalg.pinv(Theta) @ y
            
            res = minimize(L1norm, x0, method='SLSQP', bounds=bounds, constraints=constr, options={'maxiter': 10000})
            
            allsingleres_t.append(res.x)
            
        single_residuals_t = []
           
        for l,i in enumerate(allsingleres_t):
            r_temp = []
            for k,j in enumerate(singleinput):
                gidx = np.where(np.abs(allsingletruPNactivity_t[k]) >= 40)[0]
                r_temp.append(np.linalg.norm(j-np.dot(Smat,np.dot(Psimat[:,gidx], i[gidx])))/np.linalg.norm(j))
            single_residuals_t.append(r_temp)         
       
        masked_array_t = copy.deepcopy(np.array(single_residuals_t))
        for i,j in enumerate(masked_array_t):
            idx_of_min = np.argmin(j)
            masked_array_t[i][idx_of_min] = None
                
        noiselevel_result.append(len(np.where(np.isnan(np.diag(masked_array_t)))[0])/len(cidx))
else:
    noiselevel_result = np.load('./precalc/noiselevel_result3.npy')

#%% Figure S6 - CS test when noise is added

fig, ax = plt.subplots(figsize=(3,3))
plt.plot(np.arange(0,3.5,0.5), noiselevel_result, lw=3)
plt.xlabel(r'$\sigma$', fontsize=15)
plt.ylabel('% Correct', fontsize=15)
plt.ylim(0.6, 1.05)
plt.yticks([0.6, 0.7, 0.8, 0.9, 1.], fontsize=15)
plt.xticks(fontsize=15)
plt.show()


#%% Reconstruction of uPN activity from partial MBON responses

if not LOAD:
    np.random.seed(1111)
    
    partial = []
    
    for reduced_size in np.arange(50,20,-5):
        print(reduced_size)
        partial_per_size = []
    
        for oi,o in enumerate(master_odor_type[np.where(np.isnan(np.diag(masked_array1)))][:-1]):
            print(o)
            
            KCact = np.dot(Psimat, allsingletruPNactivity[oi])
            
            sample_run = 0
            
            partial_sample = []
            
            while (sample_run < 50) and not any(partial_sample):
                
                r_temp = []
                
                MBON_idx = np.sort(np.random.choice(np.shape(Smat)[0], reduced_size, replace=False))
            
                y = np.dot(Smat, KCact)
                y = y[MBON_idx]
                
                Theta_sliced = Theta[MBON_idx]
            
                bounds = scipy.optimize.Bounds(lb=-np.inf, ub=np.inf)
                constr = ({'type': 'eq', 'fun': lambda x: Theta_sliced @ x - y})
                
                x0 = np.linalg.pinv(Theta)[:,MBON_idx] @ y
                
                res = minimize(L1norm, x0, method='SLSQP', bounds=bounds, constraints=constr, options={'maxiter': 10000})
                
                for k,j in enumerate(singleinput):
                    gidx = np.where(np.abs(allsingletruPNactivity[k]) >= 40)[0]
                    r_temp.append(np.linalg.norm(j[MBON_idx]-np.dot(Smat[MBON_idx], np.dot(Psimat[:,gidx], res.x[gidx])))/np.linalg.norm(j[MBON_idx]))
                
                if np.argmin(r_temp) == oi:
                    partial_sample.append(True)
                else:
                    partial_sample.append(False)
                
                sample_run += 1
            
            if any(partial_sample):
                partial_per_size.append(True)
            else:
                partial_per_size.append(False)
                
        partial.append(partial_per_size)
        
    partial = np.array(partial)
    for i in np.arange(1, len(partial)):
        for j in np.arange(len(partial[0])):
            if partial[i][j] == True:
                if partial[i-1][j] == False:
                    partial[i-1][j] = True
else:
    partial = np.load(r'./precalc/partial_result3.npy')

correct_percentage = [1]

for p in partial:
    correct_percentage.append(Counter(p).get(True)/(len(master_odor_type) - 1))


#%% Fig 15 - Partial MBON CS test

fig, ax = plt.subplots(figsize=(3,3))
plt.plot(np.insert(np.arange(50, 20, -5), 0, 56), correct_percentage, lw=3)
plt.xlabel('$N_{MBON}$', fontsize=15)
plt.ylabel('% Correct', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


partial_p = np.insert(partial, 0, np.repeat(True, len(partial[0])), axis=0)

recoverable_odor_type = master_odor_type[np.where(np.isnan(np.diag(masked_array1)))][:-1]

fig, ax = plt.subplots(figsize=(12,3))
im = plt.imshow(partial_p, cmap='binary', aspect='equal', vmax=2)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
ax.yaxis.set_ticks_position('right')
ax.yaxis.set_label_position('right')
ax.set_yticks(np.arange(len(np.insert(np.arange(50, 20, -5), 0, 56))),
           np.insert(np.arange(50, 20, -5), 0, 56), fontsize=10)
ax.set_xticks(np.arange(len(recoverable_odor_type)), np.array(recoverable_odor_type), rotation='vertical', fontsize=10)
ax.set_yticks(np.arange(len(np.insert(np.arange(50, 20, -5), 0, 56)))-0.5, minor=True)
ax.set_xticks(np.arange(len(recoverable_odor_type))-0.5, minor=True)
ax.tick_params(which='minor', bottom=False, left=False, top=False, right=False)
ax.grid(which='minor', color='k')
for xtick, color in zip(ax.get_xticklabels(), np.array(master_odor_color)[np.where(np.isnan(np.diag(masked_array1)))][:-1]):
    xtick.set_color(color)
plt.tight_layout()
plt.show()

