import dataclasses
import logging
import os, math, random
import sys, copy
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import BertModel, BertConfig
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

class MyRandomTokenNoise:
    def __init__(self, tokenizer, noise_rate):
        self.tokenizer = tokenizer
        self.noise_rate = noise_rate

    def add_random_noise(self, inputs):
        inputs = copy.deepcopy(inputs)
        tokens = inputs['input_ids']
        tok = self.tokenizer
        for i in range(tokens.size(0)):
            for j in range(tokens.size(1)):
                if tok.decode([tokens[i][j]]).upper() in ['[CLS]', '[SEP]', '[PAD]', '</S>', '<BOS>', '<EOS>', '<S>', '<PAD>', '<SEP>', '<CLS>']:
                    #print('skipped', j, end = ' ')
                    continue
                if random.random() < self.noise_rate:
                    tokens[i][j] = min(tokens[i][j] + random.randint(1, 1000), len(tok) - 1) 
        return inputs
    
