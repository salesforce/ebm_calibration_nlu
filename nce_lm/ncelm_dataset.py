import logging
import os, copy
import time, random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from transformers.file_utils import is_tf_available
from transformers.tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_xlm_roberta import XLMRobertaTokenizer
from transformers.data.processors.glue import glue_output_modes, glue_processors #glue_convert_examples_to_features
from transformers.data.processors.utils import InputFeatures, InputExample
#from .utils import DataProcessor, InputExample, InputFeatures

logger = logging.getLogger()

@dataclass
class GlueDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(glue_processors.keys())})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

class SimpleLMDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    features: List[InputFeatures]

    def __init__(
        self,
        fn: str,
        tokenizer: PreTrainedTokenizer,
        max_len: int,
        min_len = 0,
        my_args = None,
    ):
        #self.args = args
        self.tokenizer = tokenizer
        #self.noiselm_mode, self.noiselm_partial2full_rate = noiselm_mode, noiselm_partial2full_rate
        self.my_args = my_args
        self.min_len, self.max_len = min_len, max_len
        logger.info('SimpleLMDataset fn: %s max_len: %d', fn, max_len)
        #I decided not to do caching here to avoid conflict and cnvenience
            
        pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        bos_id, eos_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token), tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        logger.info('loading from %s', fn)
        
        discard_co, features, maxlen_co = 0, [], 0
        for l in open(fn, 'r').readlines():
            l = l.strip().lstrip() #get into examples and glue_convert_examples_to_feature
            if len(l) < 2: 
                continue
            ii = tokenizer.encode(l)
            if len(ii) < min_len:
                discard_co += 1
                continue
            ii = [bos_id] + ii + [eos_id]
            if len(ii) > max_len:
                ii = ii[:max_len]
                maxlen_co += 1
            if len(ii) < max_len:
                ii = ii + [pad_id] * (max_len - len(ii))
            features.append(torch.LongTensor(ii))
        self.features = features
        logger.info('%d samples discarded', discard_co)
        logger.info('%d samples exceed max_len', maxlen_co)
        logger.info('%d examples loaded', len(self.features))
 

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    #def get_labels(self):
    #    return self.label_list

    def noiselm_convert_partial2full(self):
        assert(self.for_noiselm == True and self.my_args.noiselm_mode == 'partial2full')
        logger.info('doing convert partial2full (noiselm)')
        bos_id, eos_id, sep_id, mask_id, pad_id = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.sep_token_id, self.tokenizer.mask_token_id, self.tokenizer.pad_token_id
        self.features = []
        for i in range(len(self.features_ori)):
            ii = self.features_ori[i].tolist()
            pref = [bos_id]
            for k in range(1, len(ii)):
                if ii[k] in [bos_id, eos_id, sep_id]:
                    pref.append(ii[k])
                else:
                    if random.random() < self.my_args.noiselm_partial2full_maskrate:
                        if pref[-1] != mask_id:
                            pref.append(mask_id)
                    else:
                        pref.append(ii[k])
                if ii[k] == eos_id:
                    break
            pl = self.my_args.noiselm_partial2full_prefixlen
            if len(pref) > pl: pref = pref[:pl]
            if len(pref) < pl: pref = [pad_id] * (pl - len(pref)) + pref
            ii = (pref + ii)[:-pl]
            #am = [1 if tok != pad_id else 0 for tok in ii]
            self.features.append(torch.LongTensor(ii))
    
    def reset_iter_state(self, bz, noise_ratio = 1):
        self.bz = bz
        self.noise_ratio = noise_ratio
        self.data_idx_now, self.noise_idx_now = 0, 0
        self.data_epoch, self.noise_epoch = 0, 0
    
    def report_iter_state(self):
        logger.info('iter_state data_epoch: %d data_idx_now: %d noise_epoch: %d noise_idx_now: %d', self.data_epoch, self.data_idx_now, self.noise_epoch, self.noise_idx_now)
 
    def get_next_batch(self, mode):
        inputs = {'labels' : [], 'input_ids': [], 'attention_mask': []}
        assert(mode == 'data' or mode == 'noise')
        for kk in range(self.bz):
            if mode == 'data':
                for hh in inputs:
                    if hh == 'labels':
                        inputs[hh].append(self.features[self.data_idx_now].label)
                    else:
                        #print(getattr(self.features[self.data_idx_now], hh))
                        inputs[hh].append(torch.LongTensor(getattr(self.features[self.data_idx_now], hh)).view(1, -1))
                self.data_idx_now += 1
                
                if self.data_idx_now == len(self.features):
                    self.data_epoch += 1
                    logger.info('data features start epoch %d', self.data_epoch)
                    self.data_idx_now = 0
            
            if mode == 'noise':
                for hh in inputs:
                    if hh == 'labels':
                        inputs[hh].append(self.noise_features[self.noise_idx_now].label)
                    else:
                        inputs[hh].append(torch.LongTensor(getattr(self.noise_features[self.noise_idx_now], hh)).view(1, -1))
                self.noise_idx_now += 1
                if self.noise_idx_now == len(self.noise_features):
                    self.noise_epoch += 1
                    logger.info('noise features start epoch %d', self.noise_epoch)
                    self.noise_idx_now = 0
        
        for hh in inputs:
            if hh == 'labels':
                inputs[hh] = torch.LongTensor(inputs[hh]).cuda()
            else:
                inputs[hh] = torch.cat(inputs[hh], dim = 0).cuda()

        return inputs
   
    """
    def get_next_batch(self):
        inputs = {'labels' : [], 'input_ids': [], 'attention_mask': []}
        for kk in range(self.bz):
            for hh in inputs:
                if hh == 'labels':
                    inputs[hh].append(self.features[self.data_idx_now].label)
                else:
                    #print(getattr(self.features[self.data_idx_now], hh))
                    inputs[hh].append(torch.LongTensor(getattr(self.features[self.data_idx_now], hh)).view(1, -1))
            self.data_idx_now += 1
            
            if self.data_idx_now == len(self.features):
                self.data_epoch += 1
                logger.info('data features start epoch %d', self.data_epoch)
                self.data_idx_now = 0

            for kk_r in range(self.noise_ratio):
                for hh in inputs:
                    if hh == 'labels':
                        inputs[hh].append(self.noise_features[self.noise_idx_now].label)
                    else:
                        inputs[hh].append(torch.LongTensor(getattr(self.noise_features[self.noise_idx_now], hh)).view(1, -1))
                self.noise_idx_now += 1
                if self.noise_idx_now == len(self.noise_features):
                    self.noise_epoch += 1
                    logger.info('noise features start epoch %d', self.noise_epoch)
                    self.noise_idx_now = 0
        
        for hh in inputs:
            if hh == 'labels':
                inputs[hh] = torch.LongTensor(inputs[hh]).cuda()
            else:
                inputs[hh] = torch.cat(inputs[hh], dim = 0).cuda()

        return inputs
    """

    def load_noise_file(self, task_name, noise_fn):
        examples, idx = [], 0
        logger.info('loading noise examples from %s', noise_fn)
        discard_co = 0
        for l in open(noise_fn, 'r').readlines():
            l = l.strip().lstrip() #get into examples and glue_convert_examples_to_feature
            if len(l) < 3: continue
            str_a, str_b = None, None
            if task_name.lower() in ['qnli', 'mnli', 'qqp']:
                if (not ' <SEP> ' in l) or (len(l.split(' <SEP> ')) != 2):
                    #logger.info('warning: <SEP> not in l, discarded: %s', l)
                    discard_co += 1
                    continue
                str_a, str_b = l.split(' <SEP> ')
            else:
                str_a = l
            int_e = InputExample('noise_{}'.format(idx), str_a, str_b, 'noise')
            idx = idx + 1
            examples.append(int_e)
        logger.info('%d samples discarded', discard_co)
        logger.info('%d noise examples loaded', len(examples))
        assert(len(examples) > 10000)
        return examples


