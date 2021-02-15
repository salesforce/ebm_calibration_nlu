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

#import checklist
#from checklist.editor import Editor
#from checklist.perturb import Perturb
import sys

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


class MyGlueDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
        special_mode: str = 'normal', #normal or nce_noise
        nce_noise_file: str = None,
        for_noiselm = False,
        checklist_transform = 'none',
        my_args = None,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.for_noiselm = for_noiselm
        #self.noiselm_mode, self.noiselm_partial2full_rate = noiselm_mode, noiselm_partial2full_rate
        self.my_args = my_args
        logger.info('MyGlueDataset for_noiselm: %s', str(for_noiselm))
        if self.for_noiselm == True:
            logger.info('noiselm_mode: %s noiselm_partial2full_maskrate: %f', self.my_args.noiselm_mode, self.my_args.noiselm_partial2full_maskrate)
        else:
            logger.info('MyGlueDataset speical_mode: %s', special_mode)
        if self.for_noiselm == True:
            assert(special_mode == 'normal')
            assert(checklist_transform == 'none')
        else:
            logger.info('checklist_transform: %s', checklist_transform)
        if checklist_transform != 'none':
            assert(special_mode == 'normal')
        self.special_mode, self.checklist_transform = special_mode, checklist_transform
        self.processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}_noiselm{}_mode{}_specialmode{}".format(
                mode.value, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name, str(self.for_noiselm), str(self.my_args.noiselm_mode), str(self.special_mode)),
        )
        label_list = self.processor.get_labels()
        if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
            RobertaTokenizer,
            RobertaTokenizerFast,
            XLMRobertaTokenizer,
        ):
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        logger.info('DEBUG ignore lock now: %s', lock_path)
        #with FileLock(lock_path):
        if 1 == 1:
            logger.info('lock acquired')
            assert(args.overwrite_cache)
            if os.path.exists(cached_features_file) and not args.overwrite_cache and self.checklist_transform == 'none':
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir)
                else:
                    examples = self.processor.get_train_examples(args.data_dir)
                if limit_length is not None:
                    examples = examples[:limit_length]
                #TODO: use a noiselm version of this
                if self.checklist_transform != 'none':
                    logger.info('doing checklist_transform: %s', self.checklist_transform)
                    for idx, example in enumerate(examples):
                        if idx <= 4:
                            logger.info('idx: %d example before transform: %s', idx, str(example))
                        example.text_a, example.text_b = self.do_checklist_transform(example.text_a), self.do_checklist_transform(example.text_b)
                        if idx <= 4:
                            logger.info('idx: %d example after transform: %s', idx, str(example)) 

                if args.task_name.lower() == 'mrpc':            
                    logger.info('mrpc! replace _, and _.')
                    for idx, example in enumerate(examples):
                        example.text_a, example.text_b = example.text_a.replace(' ,', ',').replace(' .', '.').replace(" 's", "'s"), example.text_b.replace(' ,', ',').replace(' .', '.').replace(" 's", "'s")
 
                self.features = self.glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                    task = args.task_name,
                )
                if for_noiselm == True:
                    logger.info('converting features for noiselm...')
                    pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
                    bos_id, eos_id, sep_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token), tokenizer.convert_tokens_to_ids(tokenizer.eos_token), tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
                    self.feature_labels = []
                    for i in range(len(self.features)):
                        ii = self.features[i].input_ids
                        if 'label' in self.my_args.noiselm_mode:
                            assert('<LABEL{}>'.format(self.features[i].label) in tokenizer.get_vocab())                                 
                        label_id = tokenizer.convert_tokens_to_ids('<LABEL{}>'.format(self.features[i].label))
                        assert(ii[0] != bos_id)                                                                                    
                        #TODO: add label_id in
                        if self.my_args.noiselm_mode in ['labeledP2F', 'labeledH2F']:                              
                            ii = [bos_id, label_id] + ii[:-2]
                        else:                                                                       
                            ii = [bos_id] + ii[:-1]                                     
                        assert(ii[1] != bos_id)
                        #TODO: add label_id in
                        for k in range(len(ii)):
                            if ii[k] == pad_id: 
                                if self.my_args.noiselm_mode == 'H2Flabeled':
                                    ii[k] = sep_id
                                    if k + 1 < len(ii): ii[k + 1] = label_id
                                    if k + 2 < len(ii): ii[k + 2] = eos_id
                                else:
                                    ii[k] = eos_id
                                break
                        self.features[i] = torch.LongTensor(ii)
                        if i % 10000 == 0:
                            logger.info('%d converted for noiselm', i)
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

        if for_noiselm == True and (self.my_args.noiselm_mode in ['partial2full', 'labeledP2F', 'labeledH2F', 'H2Flabeled']):
            self.features_ori = copy.deepcopy(self.features)
            self.noiselm_convert_partial2full()

        if special_mode == 'nce_noise':
            ll = self.label_list
            if 'labeled' not in self.my_args.noiselm_mode:
                ll = ['noise', 'data']
            """
            noise_cached_features_file = os.path.join(
                cache_dir if cache_dir is not None else args.data_dir,
                "cached_{}_{}_{}_{}_{}".format(
                    mode.value, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
                    os.path.basename(nce_noise_file)
                ),
            )
            print('noise_cached_features_file:', noise_cached_features_file)
            lock_path = noise_cached_features_file + ".lock"
            with FileLock(lock_path):
                if os.path.exists(noise_cached_features_file) and not args.overwrite_cache:
                    start = time.time()
                    self.noise_features = torch.load(noise_cached_features_file)
                    logger.info(
                        f"Loading features from cached file {noise_cached_features_file} [took %.3f s]", time.time() - start
                    )
                else:
            """
            noise_examples = self.load_noise_file(args.task_name, nce_noise_file, label_list = ll)
            self.noise_features = self.glue_convert_examples_to_features(
                noise_examples,
                tokenizer,
                max_length=args.max_seq_length,
                label_list=ll,
                output_mode=self.output_mode,
                task = args.task_name,
            )
            #torch.save(self.noise_features, noise_cached_features_file)
            # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
            #logger.info("Saving noise_features into cached file %s]", cached_features_file)
            
            """
            if 'labeled' not in self.my_args.noiselm_mode:
                new_features = []
                #breakpoint()
                assert(ll[1] == 'data')
                for f in self.features:
                    new_features.append(InputFeatures(f.input_ids, f.attention_mask, f.token_type_ids, 1))
                self.features = new_features
            """
            for i in range(4):
                logger.info('(debug) checking data feature for nce_noise: %s', str(self.features[i]))
    
    def convert_checklist_input(
        self,
        inputs,
        given_tokenizer = None, 
        given_max_length = None,
    ):
        self.processor = glue_processors[self.args.task_name]()
        self.output_mode = glue_output_modes[self.args.task_name]
        #logger.info('label_list: %s', str(self.label_list))
        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        #examples = self.processor.get_dev_examples(self.args.data_dir)
        examples = []
        for idx, x in enumerate(inputs):
            if len(x) == 2:
                ine = InputExample('checklist_{}'.format(idx), x[0], x[1], label = self.label_list[0]) #label does not matter for check list    
            elif len(x) == 1:
                ine = InputExample('checklist_{}'.format(idx), x[0], None, label = self.label_list[0])
            elif isinstance(x, str):
                ine = InputExample('checklist_{}'.format(idx), x, None, label = self.label_list[0])
            else:
                logger.info('ERROR of len(x) in convert_checklist_input')
                sys.exit(1)
            examples.append(ine)

        features = self.glue_convert_examples_to_features(
            examples,
            self.tokenizer if given_tokenizer is None else given_tokenizer,
            max_length=self.args.max_seq_length if given_max_length is None else given_max_length,
            label_list=self.label_list,
            output_mode=self.output_mode,
            task = self.args.task_name,
            debug_print = False,
        )
        if len(features) != len(inputs):
            print('debug len(feautres) != len(inputs)')
            breakpoint()
        return features

    def do_checklist_transform(self, s):
        if s is None: return None
        if len(s) < 2: return s
        if self.checklist_transform == 'typo':
            return Perturb.add_typos(s)
        if self.checklist_transform == 'typo^2':
            return Perturb.add_typos(s, typos = 2)
    
    def noiselm_convert_partial2full(self):
        assert(self.for_noiselm == True and self.my_args.noiselm_mode in ['partial2full', 'labeledP2F', 'labeledH2F', 'H2Flabeled'])
        mode = self.my_args.noiselm_mode
        logger.info('doing convert mode: %s (noiselm)', mode)
        bos_id, eos_id, sep_id, mask_id, pad_id = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.sep_token_id, self.tokenizer.mask_token_id, self.tokenizer.pad_token_id
        self.features, discard_co = [], 0
        for i in range(len(self.features_ori)):
            ii = self.features_ori[i].tolist()
            if mode == 'labeledH2F':
                if sum([k == sep_id for k in ii]) != 1:
                    discard_co += 1
                    continue
                H2F_maskpart = random.randint(0, 1)
            if mode == 'H2Flabeled':
                if sum([k == sep_id for k in ii]) != 2:
                    discard_co += 1
                    continue
                H2F_maskpart = random.randint(0, 1) 
            start_idx = 2 if mode in ['labeledP2F', 'labeledH2F'] else 1 #we do not provide the label_id in the prefix
            pref = [bos_id]
            current_part = 0
            for k in range(start_idx, len(ii)):
                if ii[k] == sep_id:
                    current_part += 1
                if current_part == 2:
                    assert(mode == 'H2Flabeled') #have hit ... <SEP> ... <SEP> <LABEL>
                    break
                if ii[k] in [bos_id, eos_id, sep_id]:
                    pref.append(ii[k])
                else:
                    if (('H2F' not in mode) and random.random() < self.my_args.noiselm_partial2full_maskrate) or (('H2F' in mode) and current_part == H2F_maskpart):
                        if pref[-1] != mask_id:
                            pref.append(mask_id)
                    else:
                        pref.append(ii[k])
                if ii[k] == eos_id:
                    break
            
            if mode == 'H2Flabeled':
                assert(pref[-1] != sep_id and pref[-1] != eos_id)
                pref.append(eos_id)
            pl = self.my_args.noiselm_partial2full_prefixlen
            if len(pref) > pl: pref = pref[:pl]
            if len(pref) < pl: pref = [pad_id] * (pl - len(pref)) + pref
            ii = (pref + ii)[:-pl]
            #am = [1 if tok != pad_id else 0 for tok in ii]
            self.features.append(torch.LongTensor(ii))
        logger.info('discard_co: %d from %d', discard_co, len(self.features_ori))
        if discard_co > len(self.features_ori) * 0.1:
            logger.info('ERROR too many discard! something must be wrong!')
            sys.exit(1) #something could be wrong
    
    def glue_convert_examples_to_features(
        self,
        examples: Union[List[InputExample], "tf.data.Dataset"],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
        task=None,
        label_list=None,
        output_mode=None,
        debug_print = True,
    ):
        """
        Loads a data file into a list of ``InputFeatures``

        Args:
            examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
            tokenizer: Instance of a tokenizer that will tokenize the examples
            max_length: Maximum example length. Defaults to the tokenizer's max_len
            task: GLUE task
            label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
            output_mode: String indicating the output mode. Either ``regression`` or ``classification``

        Returns:
            If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
            containing the task-specific features. If the input is a list of ``InputExamples``, will return
            a list of task-specific ``InputFeatures`` which can be fed to the model.

        """
        if is_tf_available() and isinstance(examples, tf.data.Dataset):
            if task is None:
                raise ValueError("When calling glue_convert_examples_to_features from TF, the task parameter is required.")
            return _tf_glue_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)
        return self._glue_convert_examples_to_features(
            examples, tokenizer, max_length=max_length, task=task, label_list=label_list, output_mode=output_mode, debug_print = debug_print
        )

    def _glue_convert_examples_to_features(
        self,
        examples: List[InputExample],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
        task=None,
        label_list=None,
        output_mode=None,
        debug_print = True,
    ):
        if max_length is None:
            max_length = tokenizer.max_len

        if task is not None:
            processor = glue_processors[task]()
            if label_list is None:
                label_list = processor.get_labels()
                logger.info("Using label list %s for task %s" % (label_list, task))
            if output_mode is None:
                output_mode = glue_output_modes[task]
                logger.info("Using output mode %s for task %s" % (output_mode, task))

        label_map = {label: i for i, label in enumerate(label_list)}

        def label_from_example(example: InputExample) -> Union[int, float, None]:
            if example.label is None:
                return None
            if output_mode == "classification":
                return label_map[example.label]
            elif output_mode == "regression":
                return float(example.label)
            raise KeyError(output_mode)

        labels = [label_from_example(example) for example in examples]
        
        if task.lower() not in ['sst-2', 'cola']:
            new_examples = []
            discard_co = 0
            for example in examples:
                if example.text_a is None or example.text_b is None or len(example.text_a) == 0 or len(example.text_b) == 0:
                    discard_co += 1
                else:
                    new_examples.append(example)
            logger.info('%d examples discarded because of zero length, len(new_exmaples): %d', discard_co, len(new_examples))
            examples = new_examples
        
        if self.for_noiselm == True and (task.lower() not in ['sst-2', 'cola']):
            batch_encoding = tokenizer.batch_encode_plus([(example.text_a + ' ' + tokenizer.sep_token, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True,)
        else:
            batch_encoding = tokenizer.batch_encode_plus([(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True,)
        #breakpoint()

        features = []
        for i in range(len(examples)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}
            feature = InputFeatures(**inputs, label=labels[i])
            features.append(feature)
        
        if debug_print == True:
            for i, example in enumerate(examples[:2]):
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("features: %s" % features[i])

        return features

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

    def load_noise_file(self, task_name, noise_fn, label_list = None):
        examples, idx = [], 0
        #logger.info('loading noise examples from %s', noise_fn)
        discard_co = 0
        f_list = [noise_fn]
        #"""
        if 'seed1' in noise_fn:
            for seed_n in ['seed2', 'seed3', 'seed4', 'seed5']:
                new_fn = noise_fn.replace('seed1', seed_n)
                if os.path.exists(new_fn):
                    f_list.append(new_fn)
        #""" #if you want to load more nosie files, turn this on
        logger.info('detected f_list to be: %s', str(f_list))
        for fn in f_list:
            logger.info('now loading noise examples from %s', fn)
            for l in open(fn, 'r').readlines():
                l = l.strip().lstrip() #get into examples and glue_convert_examples_to_feature
                if len(l) < 3: continue
                la = 'noise'
                if 'labeled' in self.my_args.noiselm_mode:
                    tt = l.split('\t')
                    if len(tt) != 4:
                        discard_co += 1
                        continue
                    l = tt[3]
                    la = label_list[int(tt[2][6])]
                else:
                    assert('\t' not in l)
                str_a, str_b = None, None
                if task_name.lower() in ['qnli', 'mnli', 'qqp', 'rte', 'mrpc', 'sts-b', 'wnli']:
                    if (not ' <SEP> ' in l) or (len(l.split(' <SEP> ')) != 2):
                        #logger.info('warning: <SEP> not in l, discarded: %s', l)
                        discard_co += 1
                        continue
                    str_a, str_b = l.split(' <SEP> ')
                else:
                    str_a = l
                int_e = InputExample('noise_{}'.format(idx), str_a, str_b, la)
                idx = idx + 1
                examples.append(int_e)
                if idx < 5:
                    logger.info('showcasing noise InputExample: %s', str(int_e))
                if idx > 10000 and self.my_args.fast_debug > 0:
                    break
        logger.info('%d samples discarded', discard_co)
        logger.info('%d noise examples loaded', len(examples))
        assert(len(examples) > 10000)
        return examples

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list
