# Class that holds all our reserach paper's results so we can easily make data charts
import numpy as np
from copy import deepcopy

class SVIT_DATA:

    _DATA = {
        "CIFAR10" : None,
        "CIFAR100" : None,
        "Caltech256" : None
    }

    _ARCH = {
        "VIT" : None,
        "SWG" : None
    }

    _HEAD = {
        "MLP" : None,
        "SVM" : None
    }

    def __init__(self):
        
        self.DATA = deepcopy(self._DATA)

        for d in self._DATA.keys():
            self.DATA[d] = deepcopy(self._ARCH)

            for a in self._ARCH.keys():
                self.DATA[d][a] = deepcopy(self._HEAD)

                for h in self._HEAD.keys():
                    self.DATA[d][a][h] = []

        self.set_results()

    def __getitem__(self, key):
        return self.DATA[key]
    
    def __setitem__(self, key, value):
        assert len(key) == 3, "Key for __setitem__ must be a 3-list of strings"
        self.DATA[key[0]][key[1]][key[2]] = np.array(value)

    def keys(self):
        return [
            ["CIFAR10", "VIT", "MLP"],
            ["CIFAR10", "VIT", "SVM"],
            ["CIFAR10", "SWG", "MLP"],
            ["CIFAR10", "SWG", "SVM"],
            ["CIFAR100", "VIT", "MLP"],
            ["CIFAR100", "VIT", "SVM"],
            ["CIFAR100", "SWG", "MLP"],
            ["CIFAR100", "SWG", "SVM"],
            ["Caltech256", "VIT", "MLP"],
            ["Caltech256", "VIT", "SVM"],
            ["Caltech256", "SWG", "MLP"],
            ["Caltech256", "SWG", "SVM"]
        ]

    def set_results(self):
        # These are our final results from running all our experiments
        # Every index represents performance on (i+1)*(20% of training data) following training convergence
        
        self[["CIFAR10", "VIT", "MLP"]] = [0.9078, 0.9108, 0.9154, 0.923, 0.9136]
        self[["CIFAR10", "VIT", "SVM"]] = [0.9146, 0.9202, 0.9248, 0.9298, 0.9306]
        
        self[["CIFAR10", "SWG", "MLP"]] = [0.954, 0.9554, 0.9636, 0.9498, 0.9568]
        self[["CIFAR10", "SWG", "SVM"]] = [0.9596, 0.965, 0.9672, 0.9682, 0.9712]

        self[["CIFAR100", "VIT", "MLP"]] = [0.6877, 0.7195, 0.7275, 0.7396, 0.745]
        self[["CIFAR100", "VIT", "SVM"]] = [0.688, 0.7224, 0.734, 0.744, 0.7528]

        self[["CIFAR100", "SWG", "MLP"]] = [0.8006, 0.7976, 0.8063, 0.8141, 0.8195]
        self[["CIFAR100", "SWG", "SVM"]] = [0.8034, 0.8312, 0.8348, 0.8428, 0.8442]

        self[["Caltech256", "VIT", "MLP"]] = [0.8406, 0.8928, 0.9243, 0.9573, 0.9747]
        self[["Caltech256", "VIT", "SVM"]] = [0.7308, 0.8548, 0.8984, 0.926, 0.9616]

        self[["Caltech256", "SWG", "MLP"]] = [0.9264, 0.9446, 0.9656, 0.9711, 0.9881]
        self[["Caltech256", "SWG", "SVM"]] = [0.8836, 0.9384, 0.9604, 0.9736, 0.9832]

        # High level observations:
        # SVM seems to do a good job avoiding overfitting by ensuring its decision boundaries are spaced out.
        # CIFAR10/100 are smaller images with objectively less features. MLPs seem to have an overfitting issue which is not surprising
        # since MLPs are more directly learning features from the training data to form their decision boundaries. When the original feature space
        # is small (not the embedded latent space), it seems MLP is too easily capable of overfitting; smaller feature spaces more easily permit memorization 
        # instead of generalizable representation learning. Meanwhile, since the SVMs are learning features only so that they can make them linearly seperable 
        # with maximal margins, the maximal margins of the decisions are especially ammenable to generalizing onto test data since they don't converge too closely to individual datapoints
        # As a result, we see how SVMs perform better where the original feature space is smaller. 
        # Meanwhile, on Caltech256, the feature space is considerably larger 

        # Make mentions about the class counts
        
        # Make mentions about performance with more training data