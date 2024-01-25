# -*- coding: utf-8 -*-
#
# Kiri Choi, Won Kyu Kim, Changbong Hyeon
# School of Computational Sciences, Korea Institute for Advanced Study, 
# Seoul 02455, Korea
#
# This script queries neurons and fetches connections bewteen PNs, KCs, and 
# MBONs from the hemibrain dataset.
# The script will pull data from the online repository of the hemibrain dataset.
#
#
# ENTER YOUR PERSONAL TOKEN HERE ##############################################
TOKEN = ('')
###############################################################################

# FLAG TO SAVE THE FILES ######################################################
SAVE = False
###############################################################################

#%%

from neuprint import (Client, fetch_adjacencies, NeuronCriteria, fetch_neurons)

c = Client('neuprint.janelia.org', 
           dataset='hemibrain:v1.2.1', 
           token=TOKEN)

criteria_PN = NeuronCriteria(type='^.*_(.*PN)$', regex=True, inputRois=['AL(R)'], 
                               outputRois=['CA(R)'], status='Traced', cropped=False)

criteria_KC = NeuronCriteria(type='^KC.*', regex=True, inputRois=['CA(R)'], 
                             status='Traced', cropped=False)

criteria_MBON = NeuronCriteria(type='^MBON.*', regex=True, status='Traced', cropped=False, inputRois=['MB(R)'])

PN_df, PN_roi_df = fetch_neurons(criteria_PN)
KC_df, KC_roi_df = fetch_neurons(criteria_KC)
MBON_df, MBON_roi_df = fetch_neurons(criteria_MBON)

PNKCneuron_df, PNKCconn_df = fetch_adjacencies(PN_df['bodyId'], KC_df['bodyId'], rois=['CA(R)'])
KCMBONneuron_df, KCMBONconn_df = fetch_adjacencies(KC_df['bodyId'], MBON_df['bodyId'])

#%% DAN

criteria_PAM = NeuronCriteria(instance='^PAM.*', regex=True, status='Traced', cropped=False)
criteria_PPL = NeuronCriteria(instance='^PPL.*', regex=True, status='Traced', cropped=False)

PAM_df, PAM_roi_df = fetch_neurons(criteria_PAM)
PPL_df, PPL_roi_df = fetch_neurons(criteria_PPL)
PAMKCneuron_df, PAMKCconn_df = fetch_adjacencies(PAM_df['bodyId'], KC_df['bodyId'])
PPLKCneuron_df, PPLKCconn_df = fetch_adjacencies(PPL_df['bodyId'], KC_df['bodyId'])

#%%

if SAVE:
    PN_df.to_pickle(r'./data/PNKC_df.pkl')
    MBON_df.to_pickle(r'./data/MBON_df3.pkl')
    PAM_df.to_pickle(r'./data/PAM_df.pkl')
    PPL_df.to_pickle(r'./data/PPL_df.pkl')
    PNKCneuron_df.to_pickle(r'./data/neuron_PNKC_df.pkl')
    PNKCconn_df.to_pickle(r'./data/conn_PNKC_df.pkl')
    KCMBONneuron_df.to_pickle(r'./data/neuron_MBON_df3.pkl')
    KCMBONconn_df.to_pickle(r'./data/conn_MBON_df3.pkl')
    PAMKCneuron_df.to_pickle(r'./data/neuron_PAMKC_df.pkl')
    PAMKCconn_df.to_pickle(r'./data/conn_PAMKC_df.pkl')
    PPLKCneuron_df.to_pickle(r'./data/neuron_PPLKC_df.pkl')
    PPLKCconn_df.to_pickle(r'./data/conn_PPLKC_df.pkl')
    

