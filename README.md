# CSE 252D Project : Fourier Vision Transformers

### Quickstart Guide

Our main program takes 5 main arguments; other arguments can be observed in **root/src/run_experiment**
1. config_name : The config yaml file that configures this experiment
2. -D dataset_path : The location to download and/or locate the chosen dataset
3. -E experiments_path : The path to a folder to store all experiment results
4. -N experiment_name : The name of the experiment identifying this experiment
5. --download : Flag to download the dataset if it doesn't exist at the given location 

<br>

    python3 repository/src/main.py config_name -D dataset_path -E experiments_path -N experiment_name --download

Here is an example:

    python3 repository/src/main.py <config_name> -D <dataset_path> -E <experiments_path> -N <experiment_name> --download

To plot all or selected experiment results, open the notebook under 