from ntpath import realpath
from os import replace
import os
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pandas as pd

random.seed(1011)
path = os.getcwd()

# import spacy
# from SoundsLike.SoundsLike import Search

# spacy_nlp = spacy.load("en_core_web_sm")

## homophones dictionary
homophones = pd.read_json(f'{path}/homophones.json', orient = 'split')
homophones_map = dict(zip(homophones.input, homophones.output))

# def example_transform(example):
#     example["text"] = example["text"].lower()
#     return example

# https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/butter_fingers_perturbation/transformation.py
def butter_finger(text, prob=0.1):
    key_approx = {}

    key_approx["q"] = "qwasedzx"
    key_approx["w"] = "wqesadrfcx"
    key_approx["e"] = "ewrsfdqazxcvgt"
    key_approx["r"] = "retdgfwsxcvgt"
    key_approx["t"] = "tryfhgedcvbnju"
    key_approx["y"] = "ytugjhrfvbnji"
    key_approx["u"] = "uyihkjtgbnmlo"
    key_approx["i"] = "iuojlkyhnmlp"
    key_approx["o"] = "oipklujm"
    key_approx["p"] = "plo['ik"

    key_approx["a"] = "aqszwxwdce"
    key_approx["s"] = "swxadrfv"
    key_approx["d"] = "decsfaqgbv"
    key_approx["f"] = "fdgrvwsxyhn"
    key_approx["g"] = "gtbfhedcyjn"
    key_approx["h"] = "hyngjfrvkim"
    key_approx["j"] = "jhknugtblom"
    key_approx["k"] = "kjlinyhn"
    key_approx["l"] = "lokmpujn"

    key_approx["z"] = "zaxsvde"
    key_approx["x"] = "xzcsdbvfrewq"
    key_approx["c"] = "cxvdfzswergb"
    key_approx["v"] = "vcfbgxdertyn"
    key_approx["b"] = "bvnghcftyun"
    key_approx["n"] = "nbmhjvgtuik"
    key_approx["m"] = "mnkjloik"
    key_approx[" "] = " "
        
    butter_text = ""
    for letter in text:
        lcletter = letter.lower()
        if lcletter not in key_approx.keys():
            new_letter = lcletter
        else:
            if random.uniform(0,1) <= prob:
                new_letter = random.choice(key_approx[lcletter])
            else:
                new_letter = lcletter
        # go back to original case
        if lcletter != letter:
            new_letter = new_letter.upper()
        butter_text += new_letter
    return butter_text


def custom_transform(example, homophone_prob = 0.5, butter_finger_prob = 0.1):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    # raise NotImplementedError

    ##### YOUR CODE ENDS HERE ######
    ## Suppose we have a dataset with columns: more_toxic_text & less_toxic_text
    more_toxic_tokens = word_tokenize(example['more_toxic_text'])
    less_toxic_tokens = word_tokenize(example['less_toxic_text'])
    
    perturbed = ''
    for token in more_toxic_tokens:
        if homophones_map.get(token, 0):
            if random.uniform(0,1) <= homophone_prob:
                token = random.choice(homophones_map[token])
        elif random.uniform(0,1) <= butter_finger_prob:
            token = butter_finger(token, prob = butter_finger_prob)
        perturbed += token
    example['more_toxic_text'] = TreebankWordDetokenizer(perturbed)
    
    perturbed = ''
    for token in less_toxic_tokens:
        if homophones_map.get(token, 0):
            if random.uniform(0,1) <= homophone_prob:
                token = random.choice(homophones_map[token])
        elif random.uniform(0,1) <= butter_finger_prob:
            token = butter_finger(token, prob = butter_finger_prob)
        perturbed += token
    example['less_toxic_text'] = TreebankWordDetokenizer(perturbed)
    return example
                    

    # (i) closeHomophones
    # Close Homophones Swap
    # ref: https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/close_homophones_swap/transformation.py
    # doc = nlp(text)

    # spaces = [True if token.whitespace_ else False for token in doc]
    # for _ in range(max_outputs):
    #     perturbed_text = []
    #     for index, token in enumerate(doc):
    #         if random.uniform(0, 1) < corrupt_prob:
    #             try:
    #                 replacement = random.choice(
    #                     Search.closeHomophones(token.text)
    #                 )
    #                 if (
    #                     replacement.lower() != token.text.lower() # replacable?
    #                     and token.text.lower() != "a"
    #                 ):
    #                     perturbed_text.append(replacement)
    #                 else:
    #                     perturbed_text.append(token.text)
    #             except Exception:
    #                 perturbed_text.append(token.text)
    #         else:
    #             perturbed_text.append(token.text)

    #     textbf = ""
    #     for index, token in enumerate(perturbed_text):
    #         textbf += token
    #         if spaces[index]:
    #             textbf += ' '

    #     example['text'] = textbf

    # (ii) 
    return example
