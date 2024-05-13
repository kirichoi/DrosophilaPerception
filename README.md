# DrosophilaPerception

A repository containinng the scripts and data to reproduce the results presented in the manuscript titled **_Unveiling the Odor Representation in the Inner Brain of Drosophila through Compressed Sensing_**

Copyright 2024 Kiri Choi

## Description

This repository (https://github.com/kirichoi/DrosophilaPerception) provides the scripts and data to reproduce figures in our manuscript with short annotations.
Our analysis utilizes the hemibrain dataset by [Scheffer et al. 2020](https://elifesciences.org/articles/57443) to generate the connectivity matrices, part of which are included in the repository.
Few different databases not included in this repository needs to be placed along the Python scripts (at the top level) in order to run the script properly, including:

- *1-s2.0-S0960982220308587-mmc4.csv*: PN Metadata by [Bates et al. 2020](https://www.sciencedirect.com/science/article/pii/S0960982220308587) obtainable at [the journal website](https://ars.els-cdn.com/content/image/1-s2.0-S0960982220308587-mmc4.csv).
- *12915_2017_389_MOESM3_ESM.xlsx*: odor response of 31 PN classes to 17 odors by [Seki et al. 2017](https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-017-0389-z) obtainiable at [the journal website](https://static-content.springer.com/esm/art%3A10.1186%2Fs12915-017-0389-z/MediaObjects/12915_2017_389_MOESM3_ESM.xlsx).
- *elife-62576-supp1-v2.xlsx*: DAN subtype clusters by [Li et al. 2020](https://elifesciences.org/articles/62576) obtainable at [the journal website](https://cdn.elifesciences.org/articles/62576/elife-62576-supp1-v2.xlsx).

A part of these datasets is reproduced in the repository which are necessary to run these scripts. 
For these datasets, we would like to credit all the respective original authors.
Below is a short description of what each files and folders contain:

- *run_single.py*: Python script to perform analysis on single naturalistic odorants and random odor mixtures.
- *run_concentration.py*: Python script to perform analysis on natural odor mixtures and odorants at various concentrations.
- *run_MBON_raw.py*: Python script to perform analysis on raw MBON response profiles including PCA, t-SNE, etc.
- *run_nonlinear.py*: Python script to perform analysis on the nonlinear synaptic transmission.
- *run_random.py*: Python script to perform analysis on random connectivity matrices.
- *run_KCMBON_regulation.py*: Python script to perform analysis on the effects of APL-based lateral inhibition to KC activity and DAN-based modulation to KC-MBON connectivity.
- *check_connectivity.py*: Python script to plot connectivity matrices and perform condition tests.
- *import_neuprint.py*: Python script that querys the neurons and connectivities from the neuPrint database.
- *data/*: 
	- *Hallem_and_Carlson_2006_TS1.xlsx*: single odorant activity profiles translated from [Hallem and Carlson 2006](https://www.cell.com/fulltext/S0092-8674(06)00363-1).
	- *Hallem_and_Carlson_2006_TS2.xlsx*: odor mixture and concentration-dependent activity profiles translated from [Hallem and Carlson 2006](https://www.cell.com/fulltext/S0092-8674(06)00363-1).
    - *PNKC_df.pkl*: a pickled dataframe listing uPNs that project to MB calyx.
    - *MBON_df.pkl*: a pickled dataframe listing MBONs.
	- *PAM_df.pkl*: a pickled dataframe listing PAMs.
	- *PPL_df.pkl*: a pickled dataframe listing PPL1s.
	- *APL_df.pkl*: a pickled dataframe listing APL.
    - *neuron_PNKC_df.pkl*: a pickled dataframe listing uPNs and KCs with connections.
    - *conn_PNKC_df.pkl*: a pickled dataframe listing connectivity between uPNs and KCs.
	- *neuron_MBON_df.pkl*: a pickled dataframe listing KCs and MBONs with connections.
    - *conn_MBON_df.pkl*: a pickled dataframe listing connectivity between KCs and MBONs.
	- *neuron_PAMKC_df.pkl*: a pickled dataframe listing PAMs with connections.
    - *conn_PAMKC_df.pkl*: a pickled dataframe listing connectivity between PAMs and KCs.
	- *neuron_PPLKC_df.pkl*: a pickled dataframe listing PPL1s with connections.
    - *conn_PPLKC_df.pkl*: a pickled dataframe listing connectivity between PPL1s and KCs.
	- *neuron_KCAPL_df.pkl*: a pickled dataframe listing KCs and APL with connections.
    - *conn_KCAPL_df.pkl*: a pickled dataframe listing connectivity between KCs and APL.
	- *SNIC_resp.npy*: a numpy array containing SNIC response profile.
- *precalc/*: a folder containing pre-computed outputs for a number of computations that may take a long time to complete. The Python scripts prioritize using these pre-computed outputs but this behavior can be overiden by **LOAD** flag to re-run everything. ***WARNING -- THIS CAN TAKE A VERY LONG TIME!***


## Requirements

The scripts have been tested on Python 3.11.5. The following packages are necessary to run the code:

* NumPy (>=1.24.0)
* SciPy (>=1.9.3)
* matplotlib (>=3.6.2)
* pandas (>=1.5.2)
* scikit-learn (>=1.2.0)