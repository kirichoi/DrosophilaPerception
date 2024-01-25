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
# Figures related to connectivity matrices and CS condition tests are available
# in this Python script.

#%% Load datasets

import os
import numpy as np
import pandas as pd
from neuprint.utils import connection_table_to_matrix
import matplotlib
import matplotlib.pyplot as plt
import copy

os.chdir(os.path.dirname(__file__))

FAFB_glo_info = pd.read_csv('./1-s2.0-S0960982220308587-mmc4.csv')

PNKC_df = pd.read_pickle(r'./data/PNKC_df.pkl')
MBON_df = pd.read_pickle(r'./data/MBON_df3.pkl')

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

MBONbid = list(MBON_df['bodyId'])
MBONinstance = list(MBON_df['instance'])
MBONtype = list(MBON_df['type'])

neuron_PNKC_df = pd.read_pickle(r'./data/neuron_PNKC_df.pkl')
conn_PNKC_df = pd.read_pickle(r'./data/conn_PNKC_df.pkl')
neuron_MBON_df = pd.read_pickle(r'./data/neuron_MBON_df3.pkl')
conn_MBON_df = pd.read_pickle(r'./data/conn_MBON_df3.pkl')

matrix_KC = connection_table_to_matrix(conn_PNKC_df, 'bodyId')
matrix_MBON = connection_table_to_matrix(conn_MBON_df, 'bodyId')


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

MBONbid_idx = []

for i in np.sort(matrix_MBON.columns.values):
    MBONbid_idx.append(MBONbid.index(i))
    
MBON_newidx_label = np.array(MBONtype)[MBONbid_idx][matrix_MBONKCidx]


#%% Figure S1A - PN-KC connectivity matrix

fig, ax = plt.subplots(figsize=(10,20))
ax.set_yticks([])
im = plt.imshow(matrix_KC_re.T, cmap='binary', aspect='auto', interpolation='none', 
                norm=matplotlib.colors.Normalize(vmax=15))
ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_label_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_label_position('left')
plt.xticks(np.arange(len(KC_newidx_label)), np.array(KC_newidx_label), rotation='vertical', fontsize=5)
plt.show()


#%% Figure S1B - KC-MBON connectivity matrix

fig, ax = plt.subplots(figsize=(4.7,20))
ax.set_yticks([])
im = plt.imshow(matrix_MBONKC, cmap='binary', aspect='auto', interpolation='none', 
                norm=matplotlib.colors.Normalize(vmax=10))
ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_label_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_label_position('left')
plt.xticks(np.arange(len(MBON_newidx_label)), np.array(MBON_newidx_label), rotation='vertical', fontsize=5)
plt.show()

#%% Figure 10 - Relaxed coherence

Smat = np.zeros(np.shape(matrix_MBONKC.T))

for i in range(len(matrix_MBONKC.T)):
    Smat[i] = matrix_MBONKC.T[i]/np.linalg.norm(matrix_MBONKC.T[i])

Psimat = np.zeros(np.shape(matrix_KC_re.T))

for i in range(len(matrix_KC_re.T)):
    Psimat[i] = matrix_KC_re.T[i]/np.linalg.norm(matrix_KC_re.T[i])

relaxed = []

for i in range(len(Smat)):
    for j in range(len(Psimat[0])):
        relaxed.append(np.divide(np.inner(Smat[i], Psimat[:,j]), 
                                 np.multiply(np.linalg.norm(Smat[i]), np.linalg.norm(Psimat[:,j]))))
                       
fig = plt.figure(figsize=(4,3))
hist = np.histogram(relaxed, bins=100, range=(0,1), density=True)
plt.scatter(np.arange(0.005, 1, 0.01), hist[0])
plt.xlabel(r"$\mathcal{M}^{*}_{j,k, j\neq k}$", fontsize=15)
plt.ylabel('Probability', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(0, 1)
plt.yscale('log')
plt.show()
