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

## homophones dictionary
homophones = pd.read_json(f'{path}/homophones.json', orient = 'split')
homophones_map = dict(zip(homophones.input, homophones.output))


# https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/butter_fingers_perturbation/transformation.py
def butter_finger(text, prob=0.2):
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


def custom_transform(example, homophone_prob = 0.5, butter_finger_prob = 0.2):
    if example['more_toxic_text'] is not None and isinstance(example['more_toxic_text'], str):
        if example['less_toxic_text'] is not None and isinstance(example['less_toxic_text'], str):
            more_toxic_text = word_tokenize(example['more_toxic_text'])
            less_toxic_text = word_tokenize(example['less_toxic_text'])
            perturbed = []
            for token in more_toxic_text:
                if homophones_map.get(token, 0):
                    if random.uniform(0,1) <= homophone_prob:
                        token = random.choice(homophones_map[token])
                elif random.uniform(0,1) <= butter_finger_prob:
                    token = butter_finger(token, prob = butter_finger_prob)
                perturbed.append(token)
            example['more_toxic_text'] = TreebankWordDetokenizer().detokenize(perturbed)

            perturbed = []
            for token in less_toxic_text:
                if homophones_map.get(token, 0):
                    if random.uniform(0,1) <= homophone_prob:
                        token = random.choice(homophones_map[token])
                elif random.uniform(0,1) <= butter_finger_prob:
                    token = butter_finger(token, prob = butter_finger_prob)
                perturbed.append(token)
            example['less_toxic_text'] = TreebankWordDetokenizer().detokenize(perturbed)
        
    return example
