import os
ROOT_DATA_DIR = os.path.join('archive','lgg-mri-segmentation','kaggle_3m')
ROOT_STAT_DIR = './experiment_data'
MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#MEAN_STD = ([23.433, 21.251, 22.309], [34.527, 31.571, 32.965])
CLASS_WEIGHTS = ([0.0108, 0.9892])
