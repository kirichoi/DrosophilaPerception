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
# Figures related to testing DAN-based modulation of KC-MBON connectivity are 
# available in this Python script. Certain computation can take a long time and
# pre-computed array files are available.
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
import matplotlib
import matplotlib.pyplot as plt
import copy
from collections import Counter

np.random.seed(1234)

os.chdir(os.path.dirname(__file__))

FAFB_glo_info = pd.read_csv('./1-s2.0-S0960982220308587-mmc4.csv')

PNKC_df = pd.read_pickle(r'./data/PNKC_df.pkl')
MBON_df = pd.read_pickle(r'./data/MBON_df3.pkl')
PAM_df = pd.read_pickle(r'./data/PAM_df.pkl')
PPL_df = pd.read_pickle(r'./data/PPL_df.pkl')

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
PAMKCneuron_df = pd.read_pickle(r'./data/neuron_PAMKC_df.pkl')
PAMKCconn_df = pd.read_pickle(r'./data/conn_PAMKC_df.pkl')
PPLKCneuron_df = pd.read_pickle(r'./data/neuron_PPLKC_df.pkl')
PPLKCconn_df = pd.read_pickle(r'./data/conn_PPLKC_df.pkl')

matrix_KC = connection_table_to_matrix(conn_PNKC_df, 'bodyId')
matrix_MBON = connection_table_to_matrix(conn_MBON_df, 'bodyId')
matrix_PAMKC = connection_table_to_matrix(PAMKCconn_df, 'bodyId')
matrix_PPLKC = connection_table_to_matrix(PPLKCconn_df, 'bodyId')

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

matrix_PAMKC = matrix_PAMKC[KC_sorted]
matrix_PPLKC[705091769] = 0
matrix_PPLKC = matrix_PPLKC[KC_sorted]

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


Psimat = np.zeros(np.shape(matrix_KC_re.T))

for i in range(len(matrix_KC_re)):
    Psimat[:,i] = matrix_KC_re[i]/np.linalg.norm(matrix_KC_re[i])

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
    
#%% DAN clusters 

DAN_type = ["PAM11-aad (a1)", "PAM11-dd (a1)", "PAM11-nc (a1)", "PAM13-dd (b'1ap)",
            "PAM13-vd (b'1ap)", "PAM14-can (b'1m)", "PAM14-nc (b'1m)", "PAM02-ad (b'2a)",
            "PAM02-pd (b'2a)", "PAM06-can (b'2m)", "PAM06-nc (b'2m)", "PAM05-dd (b'2p)",
            "PAM05-vd (b'2p)", "PAM10-can (b1)", "PAM10-nc (b1)", "PAM09-dd (b1ped)",
            "PAM09-vd (b1ped)", "PAM04-can (b2)", "PAM04-dd (b2)", "PAM04-nc (b2)",
            "PAM03-can (b2b'2a)", "PPL101 y1pedc", "PPL102 y1", "PPL103 y2a'1",
            "PPL104 a'3", "PPL105 a'2a2", "PPL106 a3", "PAM12-dd (y3)",
            "PAM12-md (y3)", "PAM08-lc (y4)", "PAM08-md (y4)", "PAM08-nc (y4)",
            "PAM08-uc (y4)", "PAM07-can (y4<y1y2)", "PAM01-fb (y5)", "PAM01-lc (y5)",
            "PAM01-nc (y5)", "PAM01-uc (y5)", "PAM15-lcan (y5b'2a)", "PAM15-ucan (y5b'2a)"]
DAN_type = np.array(DAN_type)

if not LOAD:
    np.random.seed(1234)
    
    hemibrain_DAN_info = pd.read_excel('./elife-62576-supp1-v2.xlsx')
    
    cluster_instance = np.zeros((40, 35))
    cluster_instance[0,[2, 8, 17, 18, 19, 20, 21, 27]] = 1
    cluster_instance[1,[8, 17, 18, 19, 20, 21, 27]] = 1
    cluster_instance[2,[8, 17, 18, 19, 20, 21, 27]] = 1
    cluster_instance[3,[2, 6, 18, 28, 31, 32, 33]] = 1
    cluster_instance[4,[1, 2, 5, 11, 18, 27, 28, 29, 31, 32, 33]] = 1
    cluster_instance[5,[2, 6, 18, 31, 33]] = 1
    cluster_instance[6,[6, 10, 31, 33]] = 1
    cluster_instance[7,[2, 6, 14, 25]] = 1
    cluster_instance[8,[2, 6, 18, 19, 20, 23, 25, 27, 28, 29]] = 1
    cluster_instance[9,[2, 6, 29, 31, 33]] = 1
    cluster_instance[10,[6, 18, 23, 28, 29]] = 1
    cluster_instance[11,[2, 6, 13, 20, 29]] = 1
    cluster_instance[12,[2, 6, 29, 31, 33]] = 1
    cluster_instance[13,[7, 19, 24, 30]] = 1
    cluster_instance[14,[2, 6, 7, 8, 23, 24, 30]] = 1
    cluster_instance[15,[8, 13, 23, 30]] = 1
    cluster_instance[16,[2, 7, 23, 24, 30]] = 1
    cluster_instance[17,[7]] = 1
    cluster_instance[18,[7]] = 1
    cluster_instance[19,[7, 14, 23, 25]] = 1
    cluster_instance[20,[2, 6, 11, 29, 31]] = 1
    cluster_instance[21,[0, 1, 2, 3, 4, 8, 9, 11, 12, 15, 19, 27, 29, 30, 31, 32, 34]] = 1
    cluster_instance[22,[0, 1, 3, 4, 15, 29, 34]] = 1
    cluster_instance[23,[2, 3, 4, 10, 11, 15, 29, 31, 32]] = 1
    cluster_instance[24,[6, 10, 14, 30, 31, 32, 33]] = 1
    cluster_instance[25,[2, 8, 9, 10, 13, 30]] = 1
    cluster_instance[26,[0, 2, 3, 8, 12, 16, 25, 26, 29, 30]] = 1
    cluster_instance[27,[0, 1, 2, 3, 4, 5, 11, 29, 31, 32, 34]] = 1
    cluster_instance[28,[5, 6]] = 1
    cluster_instance[29,[6, 29, 34]] = 1
    cluster_instance[30,[5, 22, 23, 29, 34]] = 1
    cluster_instance[31,[5, 6, 12, 14, 22, 23, 25, 29, 34]] = 1
    cluster_instance[32,[5, 6, 29, 34]] = 1
    cluster_instance[33,[5, 23, 34]] = 1
    cluster_instance[34,[8, 18, 19, 22, 23, 25, 27, 28]] = 1
    cluster_instance[35,[6, 8, 22, 23, 25, 27, 28]] = 1
    cluster_instance[36,[7, 19, 22, 23, 25, 34]] = 1
    cluster_instance[37,[6, 18, 19, 22, 23, 25, 29, 34]] = 1
    cluster_instance[38,[6, 23, 25,29, 31]] = 1
    cluster_instance[39,[8, 19, 25, 27, 28, 33]] = 1

    DANs_list = [PAM_df, PPL_df]
    DANs_df = pd.concat(DANs_list)

    DAN_conn_list = [matrix_PAMKC, matrix_PPLKC]
    DAN_conn_df = pd.concat(DAN_conn_list)

    allsingleres_DAN = np.zeros((np.shape(cluster_instance)[1],len(master_odor_type),len(KC_newidx_label)))
    singleinput_DAN = np.zeros((np.shape(cluster_instance)[1],len(master_odor_type),np.shape(matrix_MBONKC)[1]))
    Smat_DAN = np.zeros((np.shape(cluster_instance)[1],np.shape(matrix_MBONKC)[1],np.shape(matrix_MBONKC)[0]))
    
    for i in range(np.shape(cluster_instance)[1]):
        print(i)
        dan_idx = np.where(cluster_instance[:,i] == 1)[0]
        active_dan = DAN_type[dan_idx]
        bid = np.array(hemibrain_DAN_info['neuPrint body ID'][np.isin(hemibrain_DAN_info['PAM subtype as shown in Figure 28-37'], active_dan)])
        try:
            bid = np.delete(bid, np.argwhere(bid==5813058050))
        except:
            pass
        DAN_sel = DAN_conn_df.loc[bid]
        DAN_sel_sum = np.sum(DAN_sel, axis=0)
        DAN_sel_sum[DAN_sel_sum == 0] = 1
        
        matrix_MBONKC_copy = copy.deepcopy(matrix_MBONKC)
        
        matrix_MBONKC_DAN = matrix_MBONKC_copy/np.array(DAN_sel_sum)[:,np.newaxis] + np.random.normal(0, matrix_MBONKC_copy/10)
        matrix_MBONKC_DAN[matrix_MBONKC_DAN<0] = 0
        
        Smat_t = np.zeros(np.shape(matrix_MBONKC_DAN.T))
    
        for j in range(len(matrix_MBONKC_DAN.T)):
            Smat_t[j] = matrix_MBONKC_DAN.T[j]/np.linalg.norm(matrix_MBONKC_DAN.T[j])
    
        Smat_DAN[i] = Smat_t
        
        Theta_t = np.matmul(Smat_t, Psimat)
        
        for m,o in enumerate(master_odor_type):
            truPNactivity = allsingletruPNactivity[m]
            KCact = np.dot(Psimat, truPNactivity)
            y = np.dot(Smat_t, KCact)
            
            singleinput_DAN[i][m] = y
            
            bounds = scipy.optimize.Bounds(lb=-np.inf, ub=np.inf)
            constr = ({'type': 'eq', 'fun': lambda x: Theta_t @ x - y})
            
            x0 = np.linalg.pinv(Theta_t) @ y
            
            res = minimize(L1norm, x0, method='SLSQP', bounds=bounds, constraints=constr, options={'maxiter': 10000})
            
            allsingleres_DAN[i][m] = res.x
    
    single_residuals_DAN = np.zeros((np.shape(cluster_instance)[1],len(master_odor_type),len(master_odor_type)))
    
    for m,n in enumerate(allsingleres_DAN):
        for l,i in enumerate(n):
            for k,j in enumerate(singleinput_DAN[m]):
                gidx = np.where(np.abs(allsingletruPNactivity[k]) >= 40)[0]
                single_residuals_DAN[m][l][k] = np.linalg.norm(j-np.dot(Smat_DAN[m],np.dot(Psimat[:,gidx], i[gidx])))/np.linalg.norm(j)
else:
    single_residuals_DAN = np.load('./precalc/single_residuals_DAN_n.npy')
    
#%% Success rate collection

cidx_DAN = []
ncidx_DAN = []
zscores_DAN = []

for r in single_residuals_DAN:
    zscores_DAN.append(-scipy.stats.zscore(r, axis=1))
    
    masked_array = copy.deepcopy(r)
    masked_array1 = copy.deepcopy(r)
    for i,j in enumerate(masked_array1):
        idx_of_min = np.argmin(j)
        masked_array1[i][idx_of_min] = None
    
    x = np.arange(len(master_odor_type))
    cidx = np.where(np.isnan(np.diag(masked_array1)))[0]
    ncidx = np.delete(x,cidx)
    
    cidx_DAN.append(cidx)
    ncidx_DAN.append(ncidx)
    
    
#%% Figure 8 - Effect of DANs on CS

single_residuals = np.load('./precalc/single_residuals3.npy')
zscores = -scipy.stats.zscore(single_residuals, axis=1)

c = 'plasma'

custom_cmap = matplotlib.cm.get_cmap(c, len(zscores_DAN))

label = []
cidx_DAN_per = []
for j,i in enumerate(cidx_DAN):
    cidx_DAN_per.append(len(i)/96)
    label.append(str(j+1))

fig, ax = plt.subplots(figsize=(12, 1.5))
ax.hlines(83/96, -1, len(cidx_DAN), lw=3, ls='dotted', color='k')
ax.scatter(np.arange(len(cidx_DAN)), cidx_DAN_per, c=np.arange(len(cidx_DAN)), cmap=c)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
ax.set_ylabel('% correct', fontsize=15)
ax.set_xlabel('DAN clusters', fontsize=15, labelpad=10)
ax.set_yticks([0.8, 0.9, 1], ['0.8', '0.9', '1.0'], fontsize=15)
ax.set_xticks(np.arange(len(cidx_DAN)), label, fontsize=12)
ticklabels = ax.get_xticklabels()
for i,j in enumerate(ticklabels):
    j.set_color(custom_cmap(i))
ax.set_ylim(0.8, 1)
ax.set_xlim(-1, 35)
plt.show()

idx = np.ceil(len(master_odor_type)/3).astype(int)

fig, ax = plt.subplots(figsize=(12, 2.5))
for j,i in enumerate(zscores_DAN):
    x = np.random.normal(0, 0.1, size=idx)
    ax.scatter(np.arange(idx)+x, np.diag(i)[:idx], color=custom_cmap(j), alpha=0.5, s=20)
ax.scatter(np.arange(idx), np.diag(zscores)[:idx], color='k', alpha=1, marker='*', s=50)
ax.set_xticks(np.arange(idx), master_odor_type[:idx], rotation=45, fontsize=10, ha='right')
for xtick, color in zip(ax.get_xticklabels(), np.array(master_odor_color)[:idx]):
    xtick.set_color(color)
ax.set_yscale('log')
ax.set_yticks([1, 10], ['$-10^{0}$', '$-10^{1}$'], fontsize=15)
ax.set_ylabel('$Z$-score', fontsize=15)
ax.set_xlim(-1, idx)
plt.show()

fig, ax = plt.subplots(figsize=(12, 2.5))
for j,i in enumerate(zscores_DAN):
    x = np.random.normal(0, 0.1, size=idx)
    ax.scatter(np.arange(idx)+x, np.diag(i)[idx:2*idx], color=custom_cmap(j), alpha=0.5, s=20)
ax.scatter(np.arange(idx), np.diag(zscores)[idx:2*idx], color='k', alpha=1, marker='*', s=50)
ax.set_xticks(np.arange(idx), master_odor_type[idx:2*idx], rotation=45, fontsize=10, ha='right')
for xtick, color in zip(ax.get_xticklabels(), np.array(master_odor_color)[idx:2*idx]):
    xtick.set_color(color)
ax.set_yscale('log')
ax.set_yticks([1, 10], ['$-10^{0}$', '$-10^{1}$'], fontsize=15)
ax.set_ylabel('$Z$-score', fontsize=15)
ax.set_xlim(-1, idx)
plt.show()

fig, ax = plt.subplots(figsize=(12, 2.5))
for j,i in enumerate(zscores_DAN):
    x = np.random.normal(0, 0.1, size=idx-2)
    ax.scatter(np.arange(idx-2)+x, np.diag(i)[2*idx:], color=custom_cmap(j), alpha=0.5, s=20)
ax.scatter(np.arange(idx-2), np.diag(zscores)[2*idx:], color='k', alpha=1, marker='*', s=50)
ax.set_xticks(np.arange(idx-2), master_odor_type[2*idx:], rotation=45, fontsize=10, ha='right')
for xtick, color in zip(ax.get_xticklabels(), np.array(master_odor_color)[2*idx:]):
    xtick.set_color(color)
ax.set_yscale('log')
ax.set_yticks([1, 10], ['$-10^{0}$', '$-10^{1}$'], fontsize=15)
ax.set_ylabel('$Z$-score', fontsize=15)
ax.set_xlim(-1, idx)
plt.show()

