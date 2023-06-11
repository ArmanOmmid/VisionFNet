# CSE 252D Project : Fourier Vision Transformers

# Repository Link
https://github.com/ArmanOmmid/VisionFNet#configuring-experiments
# Google Collab Link
https://colab.research.google.com/drive/1yPHiZWhUD0XYieyca95f-tQTwCMdwl-4?usp=sharing

## Quickstart Guide

Base Vision Transformer Architecture Credit to PyTorch
https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py

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

    python3 repository/src/main.py example -D dataset -E experiments -N example --download

Zip all the run results for keeping with:

    zip -r experiments.zip experiments

### Configuring Experiments
Refer to, modify, and create config yaml files under **root/configs** and specify the choice in the *config_name* argument.
In these yaml files, you can choose the model architecture, configurate the architecture, choose the dataset, set the hyperparameters, etc..

### The Monolothic Architecture
For simplicity and for easy comparison, we have combined our various implemented architectures into a monolothic architecture that can be configured with the various spectral layers.
All configurations below refer to this architecture **model: fvit_monolith** and from this we can represent all tested architectures and their multiscale counterparts.

### scale_factors
We can configurate the multiple patch scales with a list in **scale_factors**. Numbers in this list determine the multiple scale factors that derive from the **base_patch_size**.

### layer_config
We can configurate the network depth and layer types with a list in **layer_config**. The layer encodings are below: <br>
0 = Attention Layer <br>
1 = FViT Spectral Layer <br>
2 = GFN Spectral Layer <br>
3 = FNO Spectral Layer <br>
4 = Spectral Attention Layer <br>
-1 = Multiscale Weight Sharing FViT Spectral Layer <br>

The remaining configs deal with either attention module hyparameters, dataset choice, and general experiment hyperparameters. 

### Plotting Results
1. Locate the Jupyter Notebook at **root/src/plotter.ipynb**.
2. Specify the absolute path of your experiments folder containing all the needed experiments by assigning it as a string to the **experiments_path** variable in cell [2].
3. Optional: Include a **styles** variable to specify which experiments are being plotted and how they are styled. 
4. Run the Notebook and observe the plots.