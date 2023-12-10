import os
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pandas as pd
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
import ast
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

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


def butter_homo_transform(example, homophone_prob = 0.5, butter_finger_prob = 0.2):
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

## https://huggingface.co/facebook/wmt19-ru-en
## de_en does not work from my end
## inspired by back_translation perturbs

def en_ru_back(text): 
    if text is not None and isinstance(text, str):
        mname = "facebook/wmt19-en-ru"
        tokenizer = FSMTTokenizer.from_pretrained(mname)
        model = FSMTForConditionalGeneration.from_pretrained(mname)

        input = text
        input_ids = tokenizer.encode(input, return_tensors="pt")
        outputs = model.generate(input_ids)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        mname = "facebook/wmt19-ru-en"
        tokenizer = FSMTTokenizer.from_pretrained(mname)
        model = FSMTForConditionalGeneration.from_pretrained(mname)

        input = decoded
        input_ids = tokenizer.encode(input, return_tensors="pt")
        outputs = model.generate(input_ids)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded
    return text

    
def custom_transform(example):
    example['more_toxic_text'], example['less_toxic_text'] = (en_ru_back(example['more_toxic_text']), 
                                                              en_ru_back(example['less_toxic_text']))
    return example


## utils function 
def str_to_float(line):
    string_list = line

    result_list = ast.literal_eval(string_list)

    flattened_list = [item for sublist in result_list for item in sublist]
    return flattened_list

def map_results(df, cols):
    for col in cols[:-1]:
        df[col] = df[col].map(str_to_float)
    df[cols[-1]] = df[cols[-1]].map(ast.literal_eval)
    return df
    

def plot_loss(loss_trackers, total = 3):
    nyu_colors = ['#7b5aa6', '#330662', '#6d6d6d']
    line_style = ['-', '--']
    data_set = ['Training', 'Validation']
    aug_type = ['original', 'butter_homo', 'back_translation']
    fig, ax = plt.subplots(1,2, figsize = (22,8))
    for i, loss_tracker in enumerate(loss_trackers):
        sns.lineplot(x = range(1, len(loss_tracker['avg']) + 1), y = loss_tracker['avg'], 
                     label=f'{data_set[i//total]} Loss: {aug_type[i%total]}', color = nyu_colors[i%total], 
                     linestyle = line_style[i//total], lw = 1.5, ax = ax[i//total])
    #     std_loss = np.std(loss_tracker['loss'], axis=0)
    #     ax.fill_between(x=range(1, len(loss_tracker['avg']) + 1), y1 = loss_tracker['avg']- 2*std_loss, 
    #                     y2 = loss_tracker['avg'] + 2*std_loss, color = color, alpha = 0.5)

        ax[i//total].set_xlabel(f'{data_set[i//total]} Steps', fontsize = 15)

        ax[i//total].legend(fontsize = 12)
    fig.suptitle('Average Loss across Steps', fontsize=20)
    y_label = fig.text(0.08, 0.5, 'Marginal Ranking Loss', va='center', rotation='vertical', fontsize=15)
    save_path = '/scratch/tl2546/NLP_project/BERT/output'
    fig.savefig(f'{save_path}/MRL_plot.png', dpi = 500, bbox_inches = 'tight')
    
    
