# -*- coding: utf-8 -*-
#
# Kiri Choi, Won Kyu Kim, Changbong Hyeon
# School of Computational Sciences, Korea Institute for Advanced Study, 
# Seoul 02455, Korea
#
# This script queries neurons and fetches connections bewteen PNs, KCs, and 
# MBONs from the hemibrain dataset 
# The script will pull data from the hemibrain dataset.
# CAUTION! - THIS CAN TAKE A LONG TIME!
# Using the pickled files instead are highly recommended.
#
#
# ENTER YOUR PERSONAL TOKEN HERE ##############################################
TOKEN = ('')
###############################################################################

# FLAG TO SAVE THE FILES ######################################################
SAVE = False
###############################################################################

#%%

from neuprint import Client, fetch_adjacencies, NeuronCriteria, fetch_neurons

c = Client('neuprint.janelia.org', 
           dataset='hemibrain:v1.2.1', 
           token=TOKEN)

criteria_PNKC = NeuronCriteria(type='^.*_(.*PN)$', regex=True, 
                             inputRois=['AL(R)'], outputRois=['CA(R)'], status='Traced', cropped=False)

criteria_KC = NeuronCriteria(type='^KC.*', regex=True, 
                             inputRois=['CA(R)'], status='Traced', cropped=False)

criteria_MBON = NeuronCriteria(type='^MBON.*', regex=True, status='Traced', cropped=False)

PNKC_df, PNKC_roi_df = fetch_neurons(criteria_PNKC)
KC_df, KC_roi_df = fetch_neurons(criteria_KC)
MBON_df, MBON_roi_df = fetch_neurons(criteria_MBON)

KCneuron_df, KCconn_df = fetch_adjacencies(PNKC_df['bodyId'], KC_df['bodyId'], rois=['CA(R)'])
MBONneuron_df, MBONconn_df = fetch_adjacencies(KC_df['bodyId'], MBON_df['bodyId'])

if SAVE:
    PNKC_df.to_pickle(r'./data/PNKC_df.pkl')
    MBON_df.to_pickle(r'./data/MBON_df.pkl')
    KCneuron_df.to_pickle(r'./data/neuron_PNKC_df.pkl')
    KCconn_df.to_pickle(r'./data/conn_PNKC_df.pkl')
    MBONneuron_df.to_pickle(r'./data/neuron_MBON_df.pkl')
    MBONconn_df.to_pickle(r'./data/conn_MBON_df.pkl')

