# CSE 252D Project : Fourier Vision Transformers

## Quickstart Guide

### Running Experiments 
Our main program takes 5 main arguments; other arguments can be observed in **root/src/run_experiment**
1. *config_name* : The config yaml file that configures this experiment; referenced under **root/configs**
2. -D *data_path* : The location to download and/or locate the chosen dataset
3. -E *exp_path* : The path to a folder to store all experiment results
4. -N *exp_name* : The name of the experiment identifying this experiment
5. --download : Flag to download the dataset if it doesn't exist at the given location 

Template:

    python3 repository/src/main.py config_name -D data_path -E exp_path -N exp_name --download

Example:

    python3 repository/src/main.py example -D dataset -E experiment -N example --download

### Configuring Experiments
Refer to, modify, and create config yaml files under **root/configs** and specify these in the *config_name* argument