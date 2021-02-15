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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional
from my_lm_trainer import MyTrainer 
import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    #DataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    TrainingArguments,
    glue_tasks_num_labels,
    set_seed,
)
#from transformers import GlueDataTrainingArguments as DataTrainingArguments
#from transformers import GlueDataset
import torch
from my_glue_dataset import MyGlueDataset, GlueDataTrainingArguments
from my_data_collator import MyDataCollatorForLanguageModeling
from my_generation import generate_no_beam_search
import scheduler

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
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
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

@dataclass
class CustomArguments:
    #do_eval_calibration: bool = field(default=False, metadata={"help": "Whether to print calibration."})
    #train_from_scratch: bool = field(default=False, metadata={"help": "Train from scratch."})
    #layer_num: int = field(default=2, metadata={"help": "The hidden layer number"})
    #my_learning_rate: float = field(default=2e-5)
    #my_random_noise_rate: float = field(default=0)
    do_gen_quick: bool = field(default=False, metadata={"help": "Quickly show some samples."})
    do_gen_save: bool = field(default=False, metadata={"help": "Save the generated samples."})
    dry_run: bool = field(default = False)
    gen_save_number: int = field(default = 100)
    noiselm_mode: str = field(default = 'normal') #normal or partial2full or labeledP2F or labeledH2F (half to full) or H2Flabeled (first text, then gen label)
    noiselm_partial2full_maskrate: float = field(default = 0.5)
    noiselm_partial2full_prefixlen: int = field(default = 96)
    noiselm_labeledP2F_labelsample_temperature: float = field(default=2.0)
    bert_baseline_path: str = field(default = None)
    do_baseline_correlate: bool = field(default = False)

def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, overwrite_cache=args.overwrite_cache
        )
 
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, GlueDataTrainingArguments, TrainingArguments, CustomArguments))
    model_args, data_args, training_args, my_args = parser.parse_args_into_dataclasses()
    print('training_args:', training_args)

    """ 
    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    """

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
    logger.info('Setting seed %d', training_args.seed)
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir) 
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if model_args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)
    
    st = {'pad_token': '<PAD>', 'sep_token': '<SEP>'}
    if my_args.noiselm_mode in ['partial2full', 'labeledP2F', 'labeledH2F', 'H2Flabeled']:
        st['mask_token'] = '<M>'
    tokenizer.add_special_tokens(st)
    if 'labeled' in my_args.noiselm_mode:
        lis = []
        for kk in range(10):
            lis.append('<LABEL{}>'.format(kk))
        logger.info('adding %s to the tokenizer', str(lis))
        tokenizer.add_tokens(lis)

    model.resize_token_embeddings(len(tokenizer))
    
    #model.init_weights() #debug!

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    
    """
    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)
    """
    train_dataset = (
        MyGlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir, for_noiselm = True, my_args = my_args) #if training_args.do_train else None
    )
    eval_dataset = (
        MyGlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir, for_noiselm = True, my_args = my_args)
        #if training_args.do_eval or my_args.
        #else None
    )
    """
    test_dataset = (
        MyGlueDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir, for_noiselm = True)
        if training_args.do_predict
        else None
    )
    """
    # Get datasets

    #train_dataset = get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    #eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
    data_collator = MyDataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, mlm_probability=0
    )

    # Initialize our Trainer
    trainer = MyTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
        my_args = my_args,
        tokenizer = tokenizer,
    )
    
    #do a eval before training
    #trainer.evaluate()

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate(max_steps = 500 if (my_args.do_gen_quick or my_args.do_gen_save) else -1)

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    my_args.label_list = train_dataset.label_list
    logger.info('label_list: %s', str(my_args.label_list))

    if my_args.do_gen_quick:
        logger.info('*** Gen Quick***')
        bz = 2
        schedule = scheduler.FixedScheduler(20, is_top_p = False)
        logger.info('bz: %d sampling algorithm: %s', bz, str(schedule))
        label_acc_co, sample_co = 0, 0
        for tt in range(20):
            ww_full = torch.cat([l.view(1, -1) for l in train_dataset[bz * tt : bz * (tt + 1)]], dim = 0).cuda()
            if my_args.noiselm_mode in ['partial2full', 'labeledP2F']:
                ww_partial = ww_full[:, :my_args.noiselm_partial2full_prefixlen + 1]
            else:
                ww_partial = torch.LongTensor([tokenizer.bos_token_id] * bz).view(-1, 1).cuda()
            samples = generate_no_beam_search(model, tokenizer, ww_partial, data_args.max_seq_length, schedule, my_args = my_args)
            for i in range(bz):
                ss_ori = tokenizer.decode(ww_full[i])
                ss_ori = ss_ori.replace('<PAD> ', '') 
                ss = tokenizer.decode(samples[i])
                ss = ss.replace('<PAD> ', '') 
                if data_args.task_name.lower() in ['qnli', 'mnli', 'qqp', 'rte', 'mrpc', 'sts-b', 'wnli'] and (not '<SEP>' in ss):
                    continue
                sample_co += 1
                print('ori', i, ss_ori)
                print('sample', i, ss)
                if my_args.noiselm_mode == 'labeledP2F':
                    if samples[i][my_args.noiselm_partial2full_prefixlen + 1] == ww_full[i][my_args.noiselm_partial2full_prefixlen + 1]:
                        label_acc_co += 1
                    else:
                        print('sample', i, 'label differs with data')
        
            if my_args.noiselm_mode == 'normal':
                for i in range(bz):
                    ss = tokenizer.decode(eval_dataset[i])
                    ss = ss.replace('<PAD> ', '')
                    print('dev-data', i, ss)
        
        if my_args.noiselm_mode == 'labeledP2F':
            print('label_acc_co:', label_acc_co, 'sample_co:', sample_co)
            print('label diff rate:', (sample_co - label_acc_co) * 1.0 / sample_co)
            
    if my_args.do_gen_save:
        logger.info('*** Gen Save***')
        bz, number = 12, my_args.gen_save_number
        if my_args.do_baseline_correlate:
            bz = 8
        out_dir = training_args.output_dir + '/gen_saves/'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
            print("Directory ", out_dir ," Created ")
        schedule = scheduler.FixedScheduler(20, is_top_p = False)
        logger.info('bz: %d sampling algorithm: %s', bz, str(schedule))
        
        out_fn = out_dir + '/{}_num{}_seed{}.txt'.format(str(schedule), str(number), str(training_args.seed))
        if my_args.noiselm_mode in ['labeledP2F', 'labeledH2F']:
            out_fn = out_dir + '/{}_num{}_seed{}_labeltemp{}.txt'.format(str(schedule), str(number), str(training_args.seed), str(my_args.noiselm_labeledP2F_labelsample_temperature))
        logger.info('printing to %s', out_fn)
        
        if my_args.bert_baseline_path is not None and my_args.bert_baseline_path == 'None':
            my_args.bert_baseline_path = None
        if my_args.do_baseline_correlate:
            if my_args.bert_baseline_path is None:
                logger.info('ERROR bert_baseline_path needed')
                sys.exit(1)
            logger.info('===do_baseline_correlate, loading bert baseline model from %s', my_args.bert_baseline_path)
            num_labels = glue_tasks_num_labels[data_args.task_name]
            bert_config = AutoConfig.from_pretrained(my_args.bert_baseline_path, num_labels=num_labels, finetuning_task=data_args.task_name)
            bert_tokenizer = AutoTokenizer.from_pretrained(my_args.bert_baseline_path)
            bert_model = AutoModelForSequenceClassification.from_pretrained(my_args.bert_baseline_path)
            bert_model = bert_model.cuda()
            corr_data, corr_sample = [], []
            #bert_eval_dataset = (MyGlueDataset(data_args, tokenizer=bert_tokenizer, mode="dev", special_mode = 'normal', for_noiselm = False, my_args = my_args))

        if my_args.dry_run == False:
            outf = open(out_fn, 'w')
        else:
            logger.info('===Doing dry_run, did not output to file')
            
        get_out, b_id, traindata_idx, label_acc_co = 0, 0, 0, 0
        if my_args.noiselm_mode in ['partial2full', 'labeledP2F', 'labeledH2F', 'H2Flabeled']:
            train_dataset.noiselm_convert_partial2full()
        while get_out < number:
            ww_partial = torch.LongTensor([tokenizer.bos_token_id] * bz).view(-1, 1).cuda()
            if my_args.noiselm_mode in ['partial2full', 'labeledP2F', 'labeledH2F', 'H2Flabeled']:
                if traindata_idx + bz >= len(train_dataset):
                    logger.info('resetting traindata_idx...')
                    traindata_idx = 0
                    train_dataset.noiselm_convert_partial2full()
                ww_full = torch.cat([l.view(1, -1) for l in train_dataset[traindata_idx : traindata_idx + bz]], dim = 0).cuda()
                ww_partial = ww_full[:, :my_args.noiselm_partial2full_prefixlen + 1]
                traindata_idx += bz
            samples = generate_no_beam_search(model, tokenizer, ww_partial, data_args.max_seq_length, schedule, my_args = my_args)
            if my_args.do_baseline_correlate:
                b_samples = []

            for i in range(bz):
                
                ww = samples[i].tolist()
                
                if my_args.noiselm_mode in ['partial2full', 'H2Flabeled']:
                    ww = ww[my_args.noiselm_partial2full_prefixlen + 1:]
                if my_args.noiselm_mode in ['labeledP2F', 'labeledH2F']:
                    label_ww = ww[my_args.noiselm_partial2full_prefixlen + 1]
                    ww = ww[my_args.noiselm_partial2full_prefixlen + 2:]
                if my_args.noiselm_mode == 'H2Flabeled':
                    k = len(ww) - 1
                    while k > 0:
                        if tokenizer.decode(ww[k]).startswith('<LABEL'):
                            break
                        k = k - 1
                    if k == 0 or (not ww[k - 1] == tokenizer.sep_token_id):
                        continue
                    label_ww = ww[k]
                    ww = ww[:k-1] #remove sep and label
                
                if my_args.noiselm_mode in ['labeledH2F', 'labeledP2F']:
                    ss_ori = tokenizer.decode(ww_full[i][my_args.noiselm_partial2full_prefixlen + 2:]).replace('<PAD>', '').replace('<|endoftext|>', '').replace('  ', ' ').strip()
                    label_ori = tokenizer.decode(ww_full[i][my_args.noiselm_partial2full_prefixlen + 1].item())
                if my_args.noiselm_mode == 'H2Flabeled':
                    ss_ori = tokenizer.decode(ww_full[i][my_args.noiselm_partial2full_prefixlen + 1:]).replace('<PAD>', '').replace('<|endoftext|>', '').replace('  ', ' ').strip()
                    label_ori = ss_ori.split()[-1]
                    ss_ori = ' '.join(ss_ori.split()[:-2])
                
                if 'label' in my_args.noiselm_mode: 
                    if not label_ori.startswith('<LABEL'):
                        continue
                    assert('<LABEL' not in ss_ori)
                ss = tokenizer.decode(ww)
                ss = ss.replace('<PAD>', '').replace('<|endoftext|>', '').replace('  ', ' ').strip()
                if '\n' in ss: continue
                if (data_args.task_name.lower() not in ['qnli', 'mnli', 'qqp', 'rte', 'mrpc', 'sts-b', 'wnli']) or '<SEP>' in ss:
                    s_sample = ss
                    if my_args.noiselm_mode in ['labeledP2F', 'labeledH2F', 'H2Flabeled']:
                        s_label = tokenizer.decode(label_ww)
                        assert(s_label.startswith('<LABEL'))
                        ss = label_ori + '\t' + ss_ori + '\t' + s_label + '\t' + ss
                    if my_args.dry_run == False:
                        outf.write(ss + '\n')
                        pass
                    get_out += 1
                    if my_args.do_baseline_correlate:
                        if ' <SEP> ' in s_sample:
                            xx, xx_data = s_sample.split(' <SEP> '), ss_ori.split(' <SEP> ')
                            if len(xx) != 2: continue
                        else:
                            xx, xx_data = (s_sample,), (ss_ori,)
                        if data_args.task_name.lower() not in ['sst-2', 'cola']:
                            if len(xx) != 2 or len(xx_data) != 2:
                                continue
                        b_samples.append({'sample': xx, 'label': s_label, 'sample_data': xx_data, 'label_data': label_ori})
                    #print(i, ss);
                    #print(tokenizer.decode(ww_full[i]).replace('<PAD> ', ''))
                else:
                    continue
                if my_args.noiselm_mode in ['labeledP2F', 'labeledH2F']:
                    if samples[i][my_args.noiselm_partial2full_prefixlen + 1] == ww_full[i][my_args.noiselm_partial2full_prefixlen + 1]:
                        label_acc_co += 1
 
            b_id += 1
            if b_id % 200 == 0:
                logger.info('get_out: %d label_same_rate: %f', get_out, label_acc_co * 1.0 / get_out)
                if my_args.dry_run == False:
                    outf.flush()
                    os.fsync(outf)
            if b_id % 2000 == 0:
                logger.info('out_fn: %s', out_fn)
            
            if my_args.do_baseline_correlate and len(b_samples) > 0:
                for mode in ['data', 'sample']:
                    if mode == 'sample':
                        inputs = [t['sample'] for t in b_samples]
                        b_labels = torch.LongTensor([int(t['label'][6]) for t in b_samples])
                    if mode == 'data':
                        inputs = [t['sample_data'] for t in b_samples]
                        b_labels = torch.LongTensor([int(t['label_data'][6]) for t in b_samples])
 
                    features = eval_dataset.convert_checklist_input(inputs, given_tokenizer = bert_tokenizer, given_max_length = 128)
                    input_ids = torch.LongTensor([f.input_ids for f in features]).cuda()
                    attention_mask = torch.LongTensor([f.attention_mask for f in features]).cuda()
                    bert_model.eval()
                    outputs = bert_model(input_ids = input_ids, attention_mask = attention_mask)
                    logits = outputs[0]
                    prob = torch.softmax(logits, dim = -1).cpu().detach()
                    p_labels = torch.argmax(prob, dim = 1)
                    #probs.append(prob)
                    if mode == 'data':
                        corr_data.extend((p_labels == b_labels).detach().tolist())
                    if mode == 'sample':
                        corr_sample.extend((p_labels == b_labels).detach().tolist())
                logger.info('len: %d corr_data: %f corr_sample: %f', len(corr_data), np.mean(corr_data), np.mean(corr_sample))
        
        if my_args.dry_run == False:
            outf.close()
            comm = 'cp ' + out_fn + ' ' + out_fn + '.bak'
            logger.info('baking up output file %s', comm)
            os.system(comm)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
