# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import dataclasses
import logging
import os, math
import sys, copy
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch
import random

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import BertModel, BertConfig
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    RobertaTokenizer, RobertaForMaskedLM,
    HfArgumentParser,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)
from my_robustness import MyRandomTokenNoise
from my_trainer import MyTrainer 
from my_glue_dataset import MyGlueDataset
from my_modeling_roberta import MyRobertaForSequenceClassification, MyRobertaForNCESequenceClassification
from transformers.data.processors.utils import InputFeatures, InputExample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from my_utils import setLogger 
import checklist_utils

logger = logging.getLogger()

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class CustomArguments:
    do_eval_calibration: bool = field(default=False, metadata={"help": "Whether to print calibration."})
    do_eval_scaling_binning_calibration: bool = field(default = False)
    do_eval_noise_robustness: bool = field(default = False)
    do_eval_checklist: bool = field(default = False)
    train_from_scratch: bool = field(default=False, metadata={"help": "Train from scratch."})
    layer_num: int = field(default=2, metadata={"help": "The hidden layer number"})
    eval_steps: int = field(default = -1, metadata = {"help": "evaluate steps"})
    #my_learning_rate: float = field(default=2e-5) #just use the existing learning_rate
    my_random_noise_rate: float = field(default=0)
    fast_debug: int = field(default = 0)
    nce_noise_file: str = field(default=None)
    nce_noise_eval_file: str = field(default=None)
    nce_noise_ratio: int = field(default = 1)
    nce_lambda: float = field(default = 1)
    noiselm_mode: str = field(default='normal') 
    nce_noise_batch_size: int = field(default = 32, metadata={'help':'nce_noise_batch'})
    train_mode: str = field(default='normal') #or nce_noise
    nce_mode: str = field(default='normal') #or normal or hidden or labeled or selflabeled

    do_gen_save: bool = field(default=False, metadata={"help": "Save the generated samples."})
    dry_run: bool = field(default = False)
    gen_save_number: int = field(default = 100)
    noiselm_partial2full_maskrate: float = field(default = 0.1)
 
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, my_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, my_args = parser.parse_args_into_dataclasses()
    all_args = (model_args, data_args, training_args, my_args)
    #training_args.learning_rate = my_args.my_learning_rate

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    log_fn = training_args.output_dir + '/log_' + ('train_' if training_args.do_train else '') + ('eval_' if training_args.do_eval else '') + ('evalcalibration_' if my_args.do_eval_calibration else '') + '.txt'
    print('logger file will be set to', log_fn)
    os.system('mkdir -p ' + training_args.output_dir)
    setLogger(logger, log_fn) 
    my_args.log_fn = log_fn
    for kk in range(5): logger.info('==hostname %s', os.uname()[1])
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    model = model.cuda()
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    
    nce_noise_train_dataset, nce_noise_eval_dataset = None, None
    if my_args.train_mode == 'nce_noise' and training_args.do_train:
        # Get datasets
        nce_noise_train_dataset = (MyGlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir, special_mode = 'nce_noise', nce_noise_file = my_args.nce_noise_file, mode = 'train', for_noiselm = False, my_args = my_args))
        nce_noise_eval_dataset = (MyGlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir, special_mode = 'nce_noise', nce_noise_file = my_args.nce_noise_eval_file, mode = 'dev', for_noiselm = False, my_args = my_args))

    # Get datasets
    train_dataset = (
        MyGlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir, my_args = my_args)
    )

    eval_dataset = (MyGlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir, my_args = my_args))
    test_dataset = (
        MyGlueDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir, my_args = my_args)
        if training_args.do_predict
        else None
    )
    
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    # Initialize our Trainer
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
        tokenizer = tokenizer,
        my_args = my_args,
    )
    
    if my_args.do_gen_save:
        logger.info('*** gen_save ***')
        logger.info('mask_rate is %f', my_args.noiselm_partial2full_maskrate)
        logger.info('mask_rate is %f', my_args.noiselm_partial2full_maskrate)
        discard_co, out_co = 0, 0
        set_seed(training_args.seed) 
        import scheduler
        schedule = scheduler.FixedScheduler(10, is_top_p = False)
        bz, number = 32, my_args.gen_save_number

        out_dir = training_args.output_dir + '/gen_saves/'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
            print("Directory ", out_dir ," Created ")
        out_fn = out_dir + '/{}_num{}_seed{}.txt'.format(str(schedule), str(number), str(training_args.seed))
        logger.info('out_fn is %s', out_fn)
        
        outf = open(out_fn, 'w')
        while out_co < my_args.gen_save_number:

            dataloader = trainer.get_eval_dataloader(train_dataset) #I don't want it to be shuffled
            if training_args.seed == 42:
                dataloader = trainer.get_eval_dataloader(eval_dataset)
                logger.info('seed=42, switching to eval_dataset!')

            for step, inputs in enumerate(dataloader):
                if step % 50 == 0:
                    print('out:', out_co, 'discard:', discard_co, out_fn)
                inputs_ori = copy.deepcopy(inputs['input_ids'])
                tokens = inputs['input_ids']
                mask_lis = []
                for i in range(tokens.size(0)):
                    mm = []
                    for j in range(tokens.size(1)):
                        if tokenizer.decode([tokens[i][j]]).upper() in ['[CLS]', '[SEP]', '[PAD]', '</S>', '<BOS>', '<EOS>', '<S>', '<PAD>', '<SEP>', '<CLS>']:
                            continue
                        if random.random() < my_args.noiselm_partial2full_maskrate:
                            tokens[i][j] = tokenizer.mask_token_id
                            mm.append(j)
                    mask_lis.append(mm)
                for k, v in inputs.items():
                    inputs[k] = v.cuda()
                inputs['labels'] = None #These labels are for sentence classification
                model.eval()
                outputs = model(**inputs)
                for i in range(tokens.size(0)):
                    #print(i, 'ori:', tokenizer.decode(inputs_ori[i]).replace('<pad>', ''))
                    #print(i, 'masked:', tokenizer.decode(tokens[i]).replace('<pad>', ''))
                    for j in mask_lis[i]:
                        new_logits = schedule.transform(outputs[0][i][j].view(1, -1).cuda())
                        tokens[i][j] = torch.multinomial(torch.softmax(new_logits, dim = -1), 1)
                        #tokens[i][j] = torch.argmax(outputs[0][i][j])
                        #breakpoint()
                        #print(i, j, 'predict:', tokenizer.decode(torch.argmax(outputs[0][i][j]).item()))
                    noise_s = tokenizer.decode(tokens[i]).replace('<pad>', '').replace('</s></s>', '[SEP]').replace('</s>', '').replace('<s>', '')
                    if '[SEP]' in noise_s:
                        if len(noise_s.split('[SEP]')) != 2:
                            discard_co += 1
                            continue
                        str_a, str_b = noise_s.split('[SEP]')
                        str_a = str_a.strip().lstrip()
                        str_b = str_b.strip().lstrip()
                        out_s = str_a + ' <SEP> ' + str_b
                    else:
                        str_a = noise_s.strip().lstrip()
                        str_b = None
                        out_s = str_a
                    
                    if data_args.task_name.lower() in ['qnli', 'mnli', 'qqp', 'rte', 'mrpc', 'sts-b', 'wnli'] and str_b is None:
                        discard_co += 1
                        continue

                    #print(i, 'changed:', out_s)
                    outf.write(out_s + '\n')
                    out_co += 1
                            
        outf.close()
        comm = 'cp ' + out_fn + ' ' + out_fn + '.bak'
        logger.info('baking up output file %s', comm)
        os.system(comm)
    
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
