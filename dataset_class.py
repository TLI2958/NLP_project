import numpy as np
import pandas as pd
import os
import wordcloud

## TODO: complete dataset class just in case
class toxic_dataset():
    def __init__(self, text, toxicity, size = 10, label = None, seed = seed):
        """_summary_

        Args:
            text: comment text
            toxicity: toxic score
            size: length of training set
            label: toxic (1) or not (0). Keep it in case to calculate accuracy, etc.  
        """
        self.text = text
        self.toxicity = toxicity
        self.size = size
        self.label = label
        self.seed = seed
    
    def rate_toxicity(self, methods = None):
        """_summary_

        Args:
            methods: Defaults to None. Use the target 
            1. linear combination of target, severity toxicity, obscene, 
            identity_attach, insult, threat. see README.md for details.
            2. ?
        set self.toxicity to calculated score if applicable
        """

        pass
    
    def make_pairs(self, indices = np.zeros((1, 2), dtype=int)):
        ## terminal condition
        if len(indices) == self.size:
            return self.text[indices[:,0]], self.text[indices[:,1]]

        add_ind = np.random.randint(0, len(self.text), 
                                    size = (self.size - len(indices), 2), 
                                    random_state = self.seed)
        add_ind = np.unique(add_ind, axis = 0)
        return self.make_pairs(indices = np.hstack((indices, add_ind)))                

    def graph(self):
        """
        placeholder for wordcloud
        """
        pass