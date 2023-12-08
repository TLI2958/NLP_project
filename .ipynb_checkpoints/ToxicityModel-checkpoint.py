import numpy as np
from transformers import AutoModelForSequenceClassification
import torch
from torch import nn
import random
import pickle
import os
import copy

## ref1: https://www.kaggle.com/code/manabendrarout/pytorch-roberta-ranking-baseline-jrstc-train#Dataset
## ref2: https://www.kaggle.com/code/debarshichanda/pytorch-w-b-jigsaw-starter/notebook
## some parts hard-coded by myself ...

class ToxicityModel(nn.Module):
    def __init__(self, checkpoint='bert-base-cased', output_class = 1):
        super(ToxicityModel, self).__init__()
        self.checkpoint = checkpoint
        self.bert = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, num_labels = 2)
        self.drop = nn.Dropout(0.05)
        self.fc = nn.Linear(2, output_class)
        

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        classifier_output = out[-1]
        out = self.drop(out[-1])
        preds = self.fc(out)
        return preds, classifier_output