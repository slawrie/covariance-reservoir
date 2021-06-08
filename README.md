# UNDER CONSTRUCTION

## covariance-reservoir
----------------------
This repository contains basic scripts to generate and train models and synthetic data samples as analyzed in "Covariance-based information processing in reservoir computing systems", by S. Lawrie, R. Moreno-Bote and M. Gilson (https://doi.org/10.1101/2021.04.30.441789).

## Dependencies
The `conda`environment specifications used for this project can be found in `environment.yml`. However, not all dependencies listed there are strictly necesary.
Main required Python libraries have widespread use (matplotlib, scikit-learn, scipy, numpy, pandas, os, time), so the scripts can be ran on any environment provided those libraries are installed.

## Synthetic data
File `synthetic_data.py` contains all code relevant to produce synthetic datasets as analyzed in the paper. The instantiations we specifically analyzed will also be freely available in Zenodo, after article publication. You can otherwise contact lead author for earlier release.

## Real data
The spoken Arabic digits dataset can be freely downloaded from the UCI Machine Learning Repository website (https://archive.ics.uci.edu/ml/datasets/Spoken+Arabic+Digit). File `auxiliary_functions.py`contains functions to read the files and zero-pad them to produce samples with consistent length.

## digits_analysis_perceptron.ipynb
This Jupyter notebook displays the code used to train a mean/covariance perceptron readout, with and without reservoir. As example, we use the spoken digits dataset, located in folder `/dataset`.
