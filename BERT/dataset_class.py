## TODO: complete dataset class just in case
import numpy as np
import pandas as pd
import os

from bs4 import BeautifulSoup
from collections import defaultdict
import re
# import wordcloud

## TODO: complete dataset class just in case
class toxic_dataset():
    def __init__(self, df, text, toxicity, label, size = 10, seed = 1011):
        """
        Args:
            df (pd.DataFrame): original dataframe
            text (pd.Series): comment text
            toxicity (pd.Series): toxic score
            size: length of training set
            label: toxic (1) or not (0). Keep it in case to calculate accuracy, etc. make sure it is of the same length as text  
            e.g., toxicity_train['label'] = np.where(toxicity_train.target > 0.5, 1, 0)
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
        
    def down_sample(self, threshold = 0.001, rate = 0.1):
        """
        downsample non-toxic texts
        """
        no_toxic = self.text[self.toxicity <= threshold]
        sample_ind = self.rng.choice(no_toxic.index, size = int(len(no_toxic)*rate), 
                                     replace = False)
        no_toxic = no_toxic.loc[sample_ind]
        toxic = self.text[self.toxicity > threshold]
        self.text = pd.concat([no_toxic, toxic], axis = 0)
        self.text = self.text.sample(frac=1, random_state=self.seed)

        # update other attributes
        self.toxicity = self.toxicity[self.text.index]
        self.label = self.label[self.text.index]
        self.reset_all_indices()
        
        
    def make_pairs(self, indices = np.zeros((1, 2))):
        # terminate condition
        if len(indices) == self.size:
            self.indices = indices
            self.rearrange()
            self.reset_all_indices()
            print('paired up ...')
            return

        add_ind = self.rng.integers(0, len(self.text), 
                                    size = (self.size - len(indices), 2))
        add_ind = np.unique(add_ind, axis = 0)
        return self.make_pairs(indices = np.vstack((indices, add_ind))) if len(indices) > 1 \
            else self.make_pairs(indices = add_ind)          
    
    
    def rearrange(self):
        indices = self.indices
        self.text = pd.concat([self.text.iloc[indices[:,0]].reset_index(drop= True), 
                                  self.text.iloc[indices[:,1]].reset_index(drop=True)], 
                              axis = 1)
        self.text.columns = ['more_toxic_text', 'less_toxic_text']
        
        self.toxicity = pd.concat([self.toxicity.loc[indices[:, 0]].reset_index(drop=True),
                               self.toxicity.loc[indices[:, 1]].reset_index(drop=True)],
                              axis=1)
        self.toxicity.columns = ['toxicity_more_toxic', 'toxicity_less_toxic']

        self.label = pd.concat([self.label.loc[indices[:, 0]].reset_index(drop=True),
                            self.label.loc[indices[:, 1]].reset_index(drop=True)],
                           axis=1)
        self.label.columns = ['labels_more_toxic', 'labels_less_toxic']

        swap = self.toxicity.iloc[:,0] <= self.toxicity.iloc[:,1]
        
        self.toxicity.loc[swap, ['toxicity_more_toxic', 'toxicity_less_toxic']] = \
            self.toxicity.loc[swap, ['toxicity_less_toxic', 'toxicity_more_toxic']].values
        
        self.text.loc[swap, ['more_toxic_text', 'less_toxic_text']] = \
            self.text.loc[swap, ['less_toxic_text', 'more_toxic_text']].values
        
        self.label.loc[swap, ['labels_more_toxic', 'labels_less_toxic']] = \
            self.label.loc[swap, ['labels_less_toxic', 'labels_more_toxic']].values
        
                
    def make_dataframe(self, down_sample = False, make_pairs = False, threshold = 0.2):
        """
        Run after self.make_pairs
        """
        if down_sample:
            self.down_sample(threshold = threshold)
        if make_pairs:
            self.make_pairs()
        self.df = pd.concat([self.text, self.toxicity, self.label], axis = 1)
        print('made new dataframe ...')
        
    def graph(self):
        """
        placeholder for wordcloud
        """
        pass
    

def text_cleaning(text):
        # credit: https://www.kaggle.com/code/manabendrarout/pytorch-roberta-ranking-baseline-jrstc-train
        # util functions common in notebooks
        '''
        Cleans text into a basic form for NLP. Operations include the following:-
        1. Remove special charecters like &, #, etc
        2. Removes extra spaces
        3. Removes embedded URL links
        4. Removes HTML tags
        5. Removes emojis

        text - Text piece to be cleaned.
        '''
        template = re.compile(r'https?://\S+|www\.\S+') #Removes website links
        text = template.sub(r'', text)

        soup = BeautifulSoup(text, 'lxml') #Removes HTML tags
        only_text = soup.get_text()
        text = only_text

        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)

        text = re.sub(r"[^a-zA-Z\d]", " ", text) #Remove special Charecters
        text = re.sub(' +', ' ', text) #Remove Extra Spaces
        text = text.strip() # remove spaces at the beginning and at the end of string

        return text