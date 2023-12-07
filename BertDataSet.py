from cProfile import label
from collections import defaultdict
from json import load
import datasets
import numpy as np
from datasets import DatasetDict, load_dataset, Features, Value
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from torch.nn import functional as F
from tqdm import tqdm
import evaluate
import random
import argparse
import pickle
from utils import *
import os


class BERTDataset:
## ref https://www.kaggle.com/code/manabendrarout/pytorch-roberta-ranking-baseline-jrstc-train#Dataset
    def __init__(self, more_toxic, less_toxic, labels_more_toxic, labels_less_toxic, checkpoint='bert-base-cased'):
        self.more_toxic = more_toxic
        self.less_toxic = less_toxic
        self.labels_more_toxic = labels_more_toxic
        self.labels_less_toxic = labels_less_toxic
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.num_examples = len(self.more_toxic)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        more_toxic = str(self.more_toxic[idx])
        less_toxic = str(self.less_toxic[idx])

        tokenized_more_toxic = self.tokenizer(
            more_toxic,
            truncation=True,
            max_length=self.tokenizer.model_max_length,  
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        tokenized_less_toxic = self.tokenizer(
            less_toxic,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        ids_more_toxic = torch.tensor(tokenized_more_toxic['input_ids'], dtype=torch.long)
        mask_more_toxic = torch.tensor(tokenized_more_toxic['attention_mask'], dtype=torch.long)
        token_type_ids_more_toxic = torch.tensor(tokenized_more_toxic['token_type_ids'], dtype=torch.long)

        ids_less_toxic = torch.tensor(tokenized_less_toxic['input_ids'], dtype=torch.long)
        mask_less_toxic = torch.tensor(tokenized_less_toxic['attention_mask'], dtype=torch.long)
        token_type_ids_less_toxic = torch.tensor(tokenized_less_toxic['token_type_ids'], dtype=torch.long)

        return {
            'ids_more_toxic': ids_more_toxic,
            'mask_more_toxic': mask_more_toxic,
            'token_type_ids_more_toxic': token_type_ids_more_toxic,
            'ids_less_toxic': ids_less_toxic,
            'mask_less_toxic': mask_less_toxic,
            'token_type_ids_less_toxic': token_type_ids_less_toxic,
            'target': torch.tensor(1, dtype=torch.float),
            'labels_more_toxic': torch.tensor(self.labels_more_toxic[idx], dtype=torch.int),
            'labels_less_toxic': torch.tensor(self.labels_less_toxic[idx], dtype=torch.int)
        }
