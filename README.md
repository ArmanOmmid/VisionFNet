# CSE 252D Project : Fourier Vision Transformers

### Quickstart Guide

Our main program takes 4 main arguments:
1. config_name : The config yaml file that configures this experiment (including architecture, dataset, hyperparameters, etc...)
2. dataset_path : The location to download and/or locate the chosen dataset
3. experiments_location : The path to a folder to store all experiment results
4. experiment_name : The name of the experiment identifying this experiment
<br>
    python3 repository/src/main.py <config_name> -D <dataset_path> -E <experiments_location> -N <experiment_name> --download

To plot all or selected experiment results, open the notebook under 