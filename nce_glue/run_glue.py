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
from my_robustness import MyRandomTokenNoise
from my_trainer import MyTrainer 
from my_glue_dataset import MyGlueDataset
from my_modeling_roberta import MyRobertaForSequenceClassification, MyRobertaForNCESequenceClassification
from transformers.data.processors.utils import InputFeatures, InputExample
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
from my_utils import setLogger 
#import checklist_utils

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
    do_energy_analysis: bool = field(default = False)
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

    pcal_num_updates: int = field(default=10)
    pcal_bin_size: int = field(default=20)
    pcalloss_start_epochs: int = field(default=0)
    pcal_train: bool = field(default=False)
    pcalloss_lambda: float = field(default=1)
    pcalloss_type: str = field(default='KL')

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
    if my_args.train_mode == 'normal':
        assert('roberta' in model_args.model_name_or_path.lower())
        #model = AutoModelForSequenceClassification.from_pretrained(
        model = MyRobertaForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    
    if my_args.train_mode == 'nce_noise':
        #nce_model = MyRobertaForSequenceClassification(config)
        assert('roberta' in model_args.model_name_or_path.lower())
        model = MyRobertaForNCESequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

    if my_args.train_from_scratch:
        print('=== training from scratch! reinitilize weights')
        embed_bak = copy.deepcopy(model.bert.embeddings)
        layer_bak = copy.deepcopy(model.bert.encoder.layer)
        model.init_weights() 
        
        LL = my_args.layer_num
        print('=== applying layer_num', LL)
        # Initializing a BERT bert-base-uncased style configuration
        new_config = BertConfig(num_hidden_layers=LL)
        # Initializing a model from the bert-base-uncased style configuration
        new_bert = BertModel(new_config)

        print('=== using pretrained embedding')
        new_bert.embeddings = embed_bak
        """
        for l in range(LL):
            print('copying encoder layer', l)
            new_bert.encoder.layer[l] = layer_bak[l]
        """

        model.bert = new_bert
        model.config.num_hidden_layers = LL
    
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

    logger.info('constructing datasets (splitting eval_dataset) for calibration...')
    dataset_cal_dev1 = copy.deepcopy(eval_dataset)
    dataset_cal_dev2 = copy.deepcopy(eval_dataset)
    dataset_cal_tr = copy.deepcopy(train_dataset)
    cal_num = int(len(eval_dataset) / 2)
    dataset_cal_dev1.features = dataset_cal_dev1.features[:cal_num]
    dataset_cal_dev2.features = dataset_cal_dev2.features[-cal_num:]
    #dataset_cal_tr.features = dataset_cal_tr.features[-cal_num:]
    logger.info('setting eval_dataset to dataset_cal_dev2...')
    eval_dataset = dataset_cal_dev2

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

    print('=== random_noise_rate:', my_args.my_random_noise_rate) 
    my_noise = MyRandomTokenNoise(tokenizer, my_args.my_random_noise_rate)
    input_transform = None
    if my_args.my_random_noise_rate > 0:
        input_transform = my_noise.add_random_noise
     
    # Training
    final_evalres_savefn = None
    if training_args.do_train:
        #if my_args.train_mode == 'nce_noise':
        #    trainer.nce_train(model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None, input_transform = input_transform)
        #else:
        set_seed(training_args.seed) #set seed again before constructing suite, so that it will be the same thing when do_eval
        suite = None
        #suite = checklist_utils.construct_checklist_suite(model, tokenizer, eval_dataset, all_args)
 
        return_d = {}
        trainer.train(model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None, input_transform = input_transform, train_mode = my_args.train_mode, nce_noise_dataset = nce_noise_train_dataset, nce_noise_ratio = my_args.nce_noise_ratio, nce_noise_bz = my_args.nce_noise_batch_size, nce_mode = my_args.nce_mode, nce_noise_eval_dataset = nce_noise_eval_dataset, return_d = return_d, checklist_suite = suite, all_args = all_args)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)
        logger.info('===PRINTING EVAL_RES_LIS===')
        for eval_res in return_d['eval_res_lis']:
            logger.info(str(eval_res))
        final_evalres_savefn = training_args.output_dir + '/eval_res_save/final_eval_res.save'
        torch.save(return_d['eval_res_lis'], final_evalres_savefn)
        logger.info('eval res saved to %s', final_evalres_savefn)
    
    final_eval_results, final_checklist_eval_results = {}, {}
    final_nce_eval_results, final_nce_train_results = {}, {}

    # evaluation
    eval_results = {}
    """
    if data_args.task_name == "mnli":
        mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
        logger.info('===SWITCHING to mnli-mm for test')
        eval_dataset = GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
    """

    logger.info('seed: %d', training_args.seed)
    if training_args.do_eval:
        logger.info("*** evaluate ***")
        set_seed(training_args.seed) #set seed again before eval

        # loop to handle mnli double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        #""" #we only look at the matched dev-set for mnli (mm is mismatched)
        
        assert(len(eval_datasets) == 1) 
        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            #prediction_output = trainer.predict(test_dataset=eval_dataset)
            eval_result = trainer.evaluate(eval_dataset=eval_dataset, input_transform = input_transform)
            
            if my_args.train_mode == 'nce_noise':
                eval_nce_result = trainer.nce_evaluate(nce_noise_eval_dataset)
                final_nce_eval_results.update(eval_nce_result)
                train_nce_result = trainer.nce_evaluate(nce_noise_train_dataset, max_step = 500)
                final_nce_train_results.update(train_nce_result)
            
            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
            eval_results.update(eval_result)
            #final_eval_results['eval_acc'] = eval_result['eval_acc']
            final_eval_results.update(eval_result)
     
    if my_args.do_eval_checklist:
        logger.info('*** eval checklist***')
        set_seed(training_args.seed) #set seed again before eval
    
        suite = checklist_utils.construct_checklist_suite(model, tokenizer, eval_dataset, all_args)
        cres = checklist_utils.run_checklist_suite(model, tokenizer, eval_dataset, all_args, given_suite = suite, verbose = True)
        final_checklist_eval_results.update(cres)

        """
        if data_args.task_name.lower() == 'qqp':
            cres = checklist_utils.do_checklist_QQP(model, tokenizer, eval_dataset, all_args)
            final_checklist_eval_results.update(cres)
 
        if data_args.task_name.lower() == 'qnli':
            cres = checklist_utils.do_checklist_QNLI(model, tokenizer, eval_dataset, all_args)
            final_checklist_eval_results.update(cres)
 
        if data_args.task_name.lower() == 'sst-2':
            cres = checklist_utils.do_checklist_SST2(model, tokenizer, eval_dataset, all_args)
            final_checklist_eval_results.update(cres)
        """
         
        """
        for checklist_trans in ['typo', 'typo^2']:
            eval_checklist_dataset = MyGlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir, checklist_transform = checklist_trans, my_args = my_args)
            eval_result = trainer.evaluate(eval_dataset=eval_checklist_dataset, input_transform = None)
            for s in eval_result:
                final_checklist_eval_results['checklist_{}_{}'.format(checklist_trans, s)] = eval_result[s]
        """

    if my_args.do_eval_noise_robustness:
        # loop to handle mnli double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        set_seed(training_args.seed) #set seed again before eval

        """
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
            )
        """ #we only look at the matched dev-set for mnli (mm is mismatched)
        
        for noise_rate in [0.1, 0.2]:
            logger.info('*** eval_noise_robustness rate: %f ***', noise_rate)
            my_noise = MyRandomTokenNoise(tokenizer, noise_rate)
            input_transform = my_noise.add_random_noise
            assert(len(eval_datasets) == 1) 
            for eval_dataset in eval_datasets:
                trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
                #prediction_output = trainer.predict(test_dataset=eval_dataset)
                eval_result = trainer.evaluate(eval_dataset=eval_dataset, input_transform = input_transform)

                output_eval_file = os.path.join(
                    training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
                )
                if trainer.is_world_master():
                    with open(output_eval_file, "w") as writer:
                        logger.info("***** eval results {} *****".format(eval_dataset.args.task_name))
                        for key, value in eval_result.items():
                            logger.info("  %s = %s", key, value)
                            writer.write("%s = %s\n" % (key, value))
                if 'eval_mnli/acc' in eval_result: eval_result['eval_acc'] = eval_result['eval_mnli/acc']
                final_eval_results['randomnoise{}_eval_acc'.format(noise_rate)] = eval_result['eval_acc'] 
    
    import calibration as cal 
    from my_calibration import TScalCalibrator

    def do_cal(trainer, eval_d, do_postcal = False, do_plattbin = True, do_tscal = True, tr_d = None, ss = ''):
        prediction_output = trainer.predict(test_dataset=eval_d)
        probs_eval, labels_eval = torch.softmax(torch.FloatTensor(prediction_output.predictions), dim = -1), torch.LongTensor(prediction_output.label_ids)
        if do_postcal == False:
            ece = cal.get_ece(probs_eval.numpy(), labels_eval.numpy(), num_bins = 20)
            acc = torch.sum(torch.argmax(probs_eval, dim = -1) == labels_eval).item() * 1.0 / labels_eval.size(0)
            res = {}
            if data_args.task_name.lower() == 'cola':
                mcc_res = trainer.compute_metrics(EvalPrediction(predictions = prediction_output.predictions, label_ids = prediction_output.label_ids))
                res[ss + 'mcc'] = mcc_res['mcc']
            res.update({ss + 'acc': acc, ss + 'ece': ece})
            logger.info('cal_res: %s', str(res))
            return res
        
        prediction_output = trainer.predict(test_dataset=tr_d)
        probs_tr, labels_tr = torch.softmax(torch.FloatTensor(prediction_output.predictions), dim = -1), torch.LongTensor(prediction_output.label_ids)
        res = {}
        
        if do_plattbin == True:
            calibrator = cal.PlattBinnerMarginalCalibrator(len(probs_tr), num_bins=20)
            calibrator.train_calibration(probs_tr.numpy(), labels_tr.numpy())
            calibrated_probs_eval = torch.FloatTensor(calibrator.calibrate(probs_eval.numpy()))
            ece = cal.get_ece(calibrated_probs_eval.numpy(), labels_eval.numpy(), num_bins = 20)
            acc = torch.sum(torch.argmax(calibrated_probs_eval, dim = -1) == labels_eval).item() * 1.0 / labels_eval.size(0)
            if data_args.task_name.lower() == 'cola':
                mcc_res = trainer.compute_metrics(EvalPrediction(predictions = torch.log(calibrated_probs_eval).numpy(), label_ids = labels_eval.numpy()))
                res[ss + 'mcc'] = mcc_res['mcc']
            res.update({ss + 'plattbin_acc': acc, ss + 'plattbin_ece': ece})
        
        if do_tscal == True:
            calibrator = TScalCalibrator(num_bins=20)
            calibrator.train_calibration(probs_tr.cpu(), labels_tr.cpu())
            calibrated_probs_eval = torch.FloatTensor(calibrator.calibrate(probs_eval.cpu()))
            ece = cal.get_ece(calibrated_probs_eval.numpy(), labels_eval.numpy(), num_bins = 20)
            acc = torch.sum(torch.argmax(calibrated_probs_eval, dim = -1) == labels_eval).item() * 1.0 / labels_eval.size(0)
            if data_args.task_name.lower() == 'cola':
                mcc_res = trainer.compute_metrics(EvalPrediction(predictions = torch.log(calibrated_probs_eval).numpy(), label_ids = labels_eval.numpy()))
                res[ss + 'mcc'] = mcc_res['mcc']
            res.update({ss + 'tscal_acc': acc, ss + 'tscal_ece': ece})
            
        logger.info('cal_res: %s', str(res))
        return res
 
    if my_args.do_eval_calibration:
        logger.info("*** do calbiration ***")
        
        #if data_args.task_name.lower() == 'cola':
            #it's cola, let's do evaluate for mcc
            #res = trainer.evaluate(eval_dataset = dataset_cal_dev2)

        set_seed(training_args.seed) #set seed again before eval
        drawcal_res = trainer.eval_calibration(dataset_cal_dev2, verbose = True, fig_fn = training_args.output_dir + '/{}_calibration.pdf'.format(data_args.task_name))        
        save_fn = training_args.output_dir + '/drawcal.save'
        logger.info('saving drawcal_res to %s', save_fn)
        torch.save(drawcal_res, save_fn)

        cal_res = do_cal(trainer, dataset_cal_dev2, do_postcal = False, ss = 'cal_ori_')
        final_eval_results.update(cal_res)

    if my_args.do_eval_scaling_binning_calibration: 
        logger.info('*** do scaling_binning calibration ***') 
        set_seed(training_args.seed) 
        cal_res = {}
        cal_res.update(do_cal(trainer, dataset_cal_dev2, do_postcal = True, do_plattbin = False, do_tscal = True, tr_d = dataset_cal_dev1, ss = 'cal_dev_'))
        cal_res.update(do_cal(trainer, dataset_cal_dev2, do_postcal = True, do_plattbin = False, do_tscal = True, tr_d = dataset_cal_tr, ss = 'cal_train_'))
        logger.info('===scaling_binning_calibration %s', str(cal_res))
        final_eval_results.update(cal_res)
    
    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
            )

        for test_dataset in test_datasets:
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)
        
            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))
    
    if my_args.do_energy_analysis:
        logger.info('*** do_energy_analysis ***')
        eval_dataloader = trainer.get_eval_dataloader(dataset_cal_dev2)
        logger.info('loading baseline model...')
        if data_args.task_name.lower() == 'sst-2':
            base_model = MyRobertaForSequenceClassification.from_pretrained('./exps/glue_baseline_roberta-base/SST-2/LR2e-5BA32MAXSTEP5233WARMSTEP314/')
        if data_args.task_name.lower() == 'qnli':
            base_model = MyRobertaForSequenceClassification.from_pretrained('./exps/glue_baseline_roberta-base/QNLI/LR2e-5BA32MAXSTEP8278WARMSTEP496')
        if data_args.task_name.lower() == 'mrpc':
            base_model = MyRobertaForSequenceClassification.from_pretrained('./exps/glue_baseline_roberta-base/MRPC/LR1e-5BA16MAXSTEP2296WARMSTEP137')
        if data_args.task_name.lower() == 'mnli':
            base_model = MyRobertaForSequenceClassification.from_pretrained('./exps/glue_baseline_roberta-base/MNLI/LR2e-5BA32MAXSTEP30968WARMSTEP1858/')
        base_model = base_model.cuda()

        lis_energy, lis_logits, lis_logits_base = [], [], []
        for step, inputs in enumerate(eval_dataloader):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])
            for k, v in inputs.items():
                inputs[k] = v.cuda()    
            return_d = {}
            model.eval(); base_model.eval();
            with torch.no_grad():
                outputs = base_model(**inputs)
                lis_logits_base.append(outputs[1]) 
                inputs['special_mode'] = 'nce_noise'
                inputs['nce_mode'] = my_args.nce_mode
                inputs['return_d'] = return_d
                inputs['nce_feed_type'] = 'data'
                inputs['nce_noise_ratio'] = my_args.nce_noise_ratio 
                outputs = model(**inputs)
                lis_energy.append(return_d['nce_logits'])
                lis_logits.append(outputs[1])
        all_energy = torch.cat(lis_energy, dim = 0).view(-1)
        all_probs = torch.softmax(torch.cat(lis_logits, dim = 0), dim = -1)
        all_probs_base = torch.softmax(torch.cat(lis_logits_base, dim = 0), dim = -1)
        sorted_idx = all_energy.sort(descending = False)[1]

        save_fn = training_args.output_dir + '/dev_energy.save'
        logger.info('saving all_energy to %s', save_fn)
        torch.save({'all_energy': all_energy.cpu(), 'all_probs': all_probs.cpu(), 'all_probs_base': all_probs_base.cpu()}, save_fn)

        print('low energy:')
        for idx in sorted_idx[:10].tolist():
            print(idx, '\tenergy:', all_energy[idx].item(), 'prediction prob:', all_probs[idx].tolist(), 'prediction prob baseline:', all_probs_base[idx].tolist(), 'label:', dataset_cal_dev2[idx].label, 'text:', tokenizer.decode(dataset_cal_dev2[idx].input_ids[:100]))
        
        print('high energy:')
        for idx in sorted_idx[-10:].tolist():
            if torch.argmax(all_probs_base[idx]).item() != dataset_cal_dev2[idx].label:
                print(idx, '\tenergy:', all_energy[idx].item(), 'prediction prob:', all_probs[idx].tolist(), 'prediction prob baseline:', all_probs_base[idx].tolist(), 'label:', dataset_cal_dev2[idx].label, 'text:', tokenizer.decode(dataset_cal_dev2[idx].input_ids[:70])) 
 

    logger.info('output_dir: %s', training_args.output_dir)
    if my_args.train_mode == 'nce_noise':
        logger.info('===FINAL NCE_EVAL RESULT===')
        report_str = '[EVAL_DATA] '
        for idx in final_nce_eval_results: report_str += idx + ':' + str(final_nce_eval_results[idx])[:5] + ', '
        logger.info('%s', report_str)
        report_str = '[TRAIN_DATA] '
        for idx in final_nce_train_results: report_str += idx + ':' + str(final_nce_train_results[idx])[:5] + ', '
        logger.info('%s', report_str)
    
    """
    logger.info('===FINAL CHECKLIST_EVAL RESULTS===')
    report_str, ll = '', []
    for idx in final_checklist_eval_results: 
        if idx != 'AVG':
            report_str += idx + ':' + str(final_checklist_eval_results[idx] * 100)[:5] + '%, '
        #ll.append(final_checklist_eval_results[idx])
    logger.info('%s AVG: %s', report_str, str(final_checklist_eval_results['AVG'] * 100)[:5] + '%')
    """
 
    logger.info('===FINAL EVAL RESULTS===')
    report_str = ''
    for idx in final_eval_results: report_str += idx + ':' + str(final_eval_results[idx])[:5] + ', '
    logger.info('%s', report_str)
    
    if final_evalres_savefn is not None:
        logger.info(final_evalres_savefn)
    
    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
