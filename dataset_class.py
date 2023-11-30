import numpy as np
import pandas as pd
import os
# import wordcloud

## TODO: complete dataset class just in case
class toxic_dataset():
    def __init__(self, df, text, toxicity, size = 10, label = None, seed = 1011):
        """
        Args:
            df (pd.DataFrame): original dataframe
            text (pd.Series): comment text
            toxicity (pd.Series): toxic score
            size: length of training set
            label: toxic (1) or not (0). Keep it in case to calculate accuracy, etc. make sure it is of the same length as text  
            seed: random seed. Default to 1011
        """
        self.df = df
        self.text = text
        self.toxicity = toxicity
        self.size = size
        self.label = label
        self.seed = seed
        self.rng = np.random.default_rng(seed = self.seed)
    def rate_toxicity(self, methods = None):
        """
        Args:
            methods: Defaults to None. Use the target 
            1. linear combination of target, severity toxicity, obscene, 
            identity_attach, insult, threat. see README.md for details.
            2. ?
        """

        pass
            
    def reset_all_indices(self):
        """
        Reset all the indices
        """
        # reset index
        self.text.reset_index(drop = True, inplace = True)
        self.toxicity.reset_index(drop = True, inplace = True)
        self.label.reset_index(drop = True, inplace = True)
        
    def down_sample(self, threshold, rate = 0.1):
        """
        downsample non-toxic texts
        """
        no_toxic = self.text[self.toxicity <= threshold]
        sample_ind = self.rng.integer(0, len(no_toxic), 
                                        int(len(no_toxic) * rate), 
                                        random_state = self.seed)
        no_toxic = no_toxic.iloc[no_toxic.index[sample_ind]]
        toxic = self.text[self.toxicity > threshold]
        self.text = pd.concat([no_toxic, toxic], axis = 0)
        self.rng.shuffle(self.text)

        # update other attributes
        self.toxicity = self.toxicity[self.text.index]
        self.label = self.label[self.text.index]
        self.reset_all_indices()
        
        
    def make_pairs(self, indices = np.zeros((1, 2))):
        ## TODO: need to tackle initial condition
        # terminate condition
        if len(indices) == self.size:
            self.indices = indices
            self.rearrange()
            self.reset_all_indices()
            print('paired up ...')
            return

        add_ind = self.rng.integer(0, len(self.text), 
                                    size = (self.size - len(indices), 2), 
                                    random_state = self.seed)
        add_ind = np.unique(add_ind, axis = 0)
        return self.make_pairs(indices = np.hstack((indices, add_ind))) if len(indices) > 1 \
            else self.make_pairs(indices = add_ind)          
    
    
    def rearrange(self):
        indices = self.indices
        self.text = pd.concat([self.text.iloc[indices[:,0]], 
                                  self.text.iloc[indices[:,1]]], 
                                  columns = ['more_toxic_text', 'less_toxic_text'], axis = 1)
        self.toxicity = pd.concat([self.toxicity.iloc[indices[:,0]], 
                                  self.toxicity.iloc[indices[:,1]]], 
                                  columns = ['toxicity_more_toxic', 'toxicity_less_toxic'],
                                  axis = 1)
        self.label = pd.concat([self.label[indices.iloc[:,0]],
                                   self.label[indices.iloc[:,1]]], 
                                   columns = ['labels_more_toxic', 'labels_less_toxic'],
                                   axis = 1)     
        swap = self.toxicity.iloc[:,0] <= self.toxicity.iloc[:,1]
        
        self.toxicity.loc[swap, ['toxicity_more_toxic', 'toxicity_less_toxic']] = \
            self.toxicity.loc[swap, ['toxicity_less_toxic', 'toxicity_more_toxic']].values
        
        self.text.loc[swap, ['more_toxic_text', 'less_toxic_text']] = \
            self.text.loc[swap, ['less_toxic_text', 'more_toxic_text']].values
        
        self.label.loc[swap, ['label_more_toxic', 'label_less_toxic']] = \
            self.label.loc[swap, ['label_less_toxic', 'label_more_toxic']].values
        
    def make_dataframe(self, down_sample = False, make_pairs = False):
        """
        Run after self.make_pairs
        """
        if down_sample:
            self.down_sample()
        if make_pairs:
            self.make_pairs()
        self.df = pd.concat([self.text, self.toxicity, self.label], axis = 1)
        print('made new dataframe ...')
        
        
    def graph(self):
        """
        placeholder for wordcloud
        """
        pass