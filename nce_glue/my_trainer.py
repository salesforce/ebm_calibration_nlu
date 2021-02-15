import json
import logging
import math
import os, copy
import random
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import collections
from collections import defaultdict, OrderedDict

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from tqdm.auto import tqdm, trange
tqdm.get_lock().locks = []

from transformers.data.data_collator import DataCollator, DefaultDataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, PredictionOutput, TrainOutput
from transformers.training_args import TrainingArguments#, is_tpu_available

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#import checklist_utils

try:
    from apex import amp

    _has_apex = True
except ImportError:
    _has_apex = False


def is_apex_available():
    return _has_apex

"""
if is_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
"""

try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


def is_tensorboard_available():
    return _has_tensorboard


try:
    import wandb

    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn("W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.")
    else:
        _has_wandb = False if os.getenv("WANDB_DISABLED") else True
except ImportError:
    _has_wandb = False


def is_wandb_available():
    return _has_wandb


logger = logging.getLogger()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()

class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def get_tpu_sampler(dataset: Dataset):
    if xm.xrt_world_size() <= 1:
        return RandomSampler(dataset)
    return DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())

def setup_bins(bin_size, adaptive=False):
    if adaptive:
        return "TODO"
    else:
        return np.linspace(0,1,bin_size+1)

class MyTrainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch,
    optimized for Transformers.
    """

    model: PreTrainedModel
    args: TrainingArguments
    data_collator: DataCollator
    train_dataset: Optional[Dataset]
    eval_dataset: Optional[Dataset]
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    prediction_loss_only: bool
    tb_writer: Optional["SummaryWriter"] = None
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None
    global_step: Optional[int] = None
    epoch: Optional[float] = None

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        prediction_loss_only=False,
        tb_writer: Optional["SummaryWriter"] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
        tokenizer = None,
        my_args = None,
    ):
        """
        Trainer is a simple but feature-complete training and eval loop for PyTorch,
        optimized for Transformers.

        Args:
            prediction_loss_only:
                (Optional) in evaluation and prediction, only return the loss
        """
        self.model = model.to(args.device)
        self.tokenizer = tokenizer
        self.args = args
        self.my_args = my_args
        if data_collator is not None:
            self.data_collator = data_collator
        else:
            self.data_collator = DefaultDataCollator()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.prediction_loss_only = prediction_loss_only
        self.optimizers = optimizers
        if tb_writer is not None:
            self.tb_writer = tb_writer
        elif is_tensorboard_available() and self.is_world_master():
            self.tb_writer = SummaryWriter(log_dir=self.args.logging_dir)
        if not is_tensorboard_available():
            logger.warning(
                "You are instantiating a Trainer but Tensorboard is not installed. You should consider installing it."
            )
        if is_wandb_available():
            self._setup_wandb()
        else:
            logger.info(
                "You are instantiating a Trainer but W&B is not installed. To use wandb logging, "
                "run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface."
            )
        set_seed(self.args.seed)
        # Create output directory if needed
        if self.is_world_master():
            os.makedirs(self.args.output_dir, exist_ok=True)
        """
        if is_tpu_available():
            # Set an xla_device flag on the model's config.
            # We'll find a more elegant and not need to do this in the future.
            self.model.config.xla_device = True
        """

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        """
        if is_tpu_available():
            train_sampler = get_tpu_sampler(self.train_dataset)
        else:
        """
        train_sampler = (
            RandomSampler(self.train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(self.train_dataset)
        )

        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator.collate_batch,
        )

        return data_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        """
        if is_tpu_available():
            sampler = SequentialDistributedSampler(
                eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        """
        if self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(eval_dataset)
        else:
            sampler = SequentialSampler(eval_dataset)

        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator.collate_batch,
        )

        return data_loader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        # We use the same batch_size as for eval.
        """
        if is_tpu_available():
            sampler = SequentialDistributedSampler(
                test_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        """
        if self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(test_dataset)
        else:
            sampler = SequentialSampler(test_dataset)

        data_loader = DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator.collate_batch,
        )

        return data_loader

    def get_optimizers(
        self, num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well.
        If you want to use something else, you can pass a tuple in the Trainer's init,
        or override this method in a subclass.
        """
        if self.optimizers is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def _setup_wandb(self):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can override this method to customize the setup if needed.  Find more information at https://docs.wandb.com/huggingface
        You can also override the following environment variables:

        Environment:
            WANDB_WATCH:
                (Optional, ["gradients", "all", "false"]) "gradients" by default, set to "false" to disable gradient logging
                or "all" to log gradients and parameters
            WANDB_PROJECT:
                (Optional): str - "huggingface" by default, set this to a custom string to store results in a different project
            WANDB_DISABLED:
                (Optional): boolean - defaults to false, set to "true" to disable wandb entirely
        """
        logger.info('Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"')
        wandb.init(project=os.getenv("WANDB_PROJECT", "huggingface"), config=vars(self.args))
        # keep track of model topology and gradients
        if os.getenv("WANDB_WATCH") != "false":
            wandb.watch(
                self.model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, self.args.logging_steps)
            )

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get num of examples from a DataLoader, by accessing its Dataset.
        """
        return len(dataloader.dataset)
    
    def nce_evaluate(self, nce_noise_eval_dataset, max_step = -1):
        nce_noise_eval_dataset.reset_iter_state(self.my_args.nce_noise_batch_size, noise_ratio = self.my_args.nce_noise_ratio)
        assert(nce_noise_eval_dataset.data_epoch == 0)
        model = self.model
        eval_stat_d = {'energy_data_mean': [], 'energy_noise_mean': [], 'energy_data_max': [], 'energy_noise_min': [], 'energy_disc_acc': [], 'nce_loss': []}
        
        step = 0
        while nce_noise_eval_dataset.data_epoch == 0:
            stat_d_tmp = self.nce_forward(model, nce_noise_eval_dataset, do_train = False)
            for idx in eval_stat_d:
                eval_stat_d[idx].append(np.mean(stat_d_tmp[idx]))
            step += 1
            if max_step >= 0 and step > max_step:
                break
        
        for idx in eval_stat_d: eval_stat_d[idx] = np.mean(eval_stat_d[idx])
        logger.info('nce_evaluate result %s', str(eval_stat_d))
        return eval_stat_d
    
    def nce_forward(self, model, nce_noise_dataset, do_train = False):
        stat_d_tmp = {'energy_data_mean': [], 'energy_noise_mean': [], 'energy_data_max': -100000, 'energy_noise_min': 100000, 'nce_loss': []}
        for kk in range(self.my_args.nce_noise_ratio + 1):
            feed_type = 'data' if kk == 0 else 'noise'
            nce_inputs = nce_noise_dataset.get_next_batch(feed_type)
            if do_train == True: 
                model.train() 
            else: 
                model.eval()
            nce_inputs['special_mode'] = 'nce_noise'
            nce_inputs['nce_mode'] = self.my_args.nce_mode
            return_d = {}
            nce_inputs['return_d'] = return_d
            nce_inputs['nce_feed_type'] = feed_type
            nce_inputs['nce_noise_ratio'] = self.my_args.nce_noise_ratio
            #breakpoint()
            #outputs = model(**nce_inputs, special_mode == 'nce_noise', nce_mode = 'normal')
            outputs = model(**nce_inputs)
            nce_loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            #nce_loss_tmp.append(nce_loss.item())
            stat_d_tmp['nce_loss'].append(nce_loss.item())
            nce_loss = nce_loss * self.my_args.nce_lambda
            if do_train == True:
                nce_loss.backward(retain_graph = True) 

            if 'energy_data_max' in return_d:
                stat_d_tmp['energy_data_max'] = max(stat_d_tmp['energy_data_max'], return_d['energy_data_max'])
                return_d['energy_data_max'] = None
            if 'energy_noise_min' in return_d:
                stat_d_tmp['energy_noise_min'] = min(stat_d_tmp['energy_noise_min'], return_d['energy_noise_min'])
                return_d['energy_noise_min'] = None
            for idx in stat_d_tmp:
                if idx in return_d and return_d[idx] is not None:
                    stat_d_tmp[idx].append(return_d[idx])
        
        #print('energy_data_max:', stat_d_tmp['energy_data_max'], 'energy_noise_min:', stat_d_tmp['energy_noise_min'])
        #stat_d_tmp['energy_data_max'], stat_d_tmp['energy_noise_min'] = None, None
        stat_d_tmp['energy_disc_acc'] = 1.0 if (stat_d_tmp['energy_data_max'] < stat_d_tmp['energy_noise_min']) else 0.0
        return stat_d_tmp
    
    def eval_calibration(self, eval_dataset, verbose = False, fig_fn = None):
        results_d = {}
        acc_thres = [0]
        acc_chunk_size = 0.05
        for i in range(math.floor(1 / acc_chunk_size) + 2): acc_thres.append(acc_thres[-1] + acc_chunk_size) #each chunk refers to error in acc_thres[i] .. acc_thres[i+1]
        if verbose: logger.info('acc_thres: %s', str(acc_thres))
        
        #for eval_dataset in eval_datasets:
        #trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
        prediction_output = self.predict(test_dataset=eval_dataset)
        logits, label_ids = torch.FloatTensor(prediction_output.predictions), torch.LongTensor(prediction_output.label_ids)
        label_num = logits.size(1)
        acc = torch.sum(torch.argmax(logits, dim = -1) == label_ids).item() * 1.0 / label_ids.size(0)
        if verbose: logger.info('acc: %f', acc)
        
        probs = torch.softmax(logits, dim = -1)
        all_probs = []
        acc_bins = [[] for a in acc_thres]
        prob_bins = [[] for a in acc_thres]
        all_num = 0
        for i in range(probs.size(0)):
            p_prob, p_id = torch.max(probs[i]).item(), torch.argmax(probs[i]).item()
            all_probs.append(p_prob)
            for j in range(label_num):
                all_num = all_num + 1
                chunk_id = math.floor(probs[i][j] / acc_chunk_size)
                assert(probs[i][j] >= acc_thres[chunk_id] and probs[i][j] <= acc_thres[chunk_id + 1])
                if j == label_ids[i].item(): 
                    acc_bins[chunk_id].append(1.0)
                else:
                    acc_bins[chunk_id].append(0.0)
                prob_bins[chunk_id].append(probs[i][j])
        
        #computing ece
        ece = 0
        for i in range(len(acc_bins)):
            if len(acc_bins[i]) > 0:
                ece += len(acc_bins[i]) * 1.0 / all_num * abs(np.mean(acc_bins[i]) - np.mean(prob_bins[i]))
        
        if verbose: 
            logger.info('===ECE: %f', ece)
            acc_bins_mean = [np.mean(m) if len(m) > 0 else -0.1 for m in acc_bins]

            fig = plt.figure(figsize = (10, 5))
            ax = plt.subplot(2, 1, 1)
            x = [a + acc_chunk_size / 2.0 for a in acc_thres]
            
            ax.plot(x, acc_bins_mean, 'x')
            ax.plot(x, x, 'x')
            for i in range(len(acc_bins)):
                ax.text(x[i], acc_bins_mean[i], str(len(acc_bins[i])), fontsize = 6)
                ax.text(x[i], acc_bins_mean[i] - 0.025, 'x:'+str(x[i])[:5], fontsize = 3)
            
            ax = plt.subplot(2, 1, 2)
            ax.plot(sorted(all_probs)) 
            ax.legend(['sorted predict max prob'])

            #fig_fn = training_args.output_dir + '/{}_calibration.pdf'.format(data_args.task_name)
            plt.savefig(fig_fn)
            logger.info('ece figure saved to %s', fig_fn)
            plt.close()
        
        results_d['acc_bins'] = acc_bins
        results_d['prob_bins'] = prob_bins
        results_d['acc_thres'] = acc_thres
        results_d['ece'] = ece

        return results_d

    def train(self, model_path: Optional[str] = None, input_transform = None, train_mode = 'normal', nce_noise_dataset = None, nce_noise_ratio = None, nce_noise_bz = None, nce_mode = 'normal', nce_noise_eval_dataset = None, return_d = None, checklist_suite = None, all_args = None):
        """
        Main training entry point.

        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs
        
        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        """
        if is_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
        """
        total_train_batch_size = (
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
        )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info('  logging_steps = %d', self.args.logging_steps)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        
        if train_mode == 'nce_noise':
            logger.info('nce_noise mode: %s bz: %d ratio: %d lambda: %f', nce_mode, nce_noise_bz, nce_noise_ratio, self.my_args.nce_lambda)
            self.nce_noise_ratio = nce_noise_ratio
            self.nce_mode = nce_mode
            logger.info('resetting nce-noise_dataset')
            nce_noise_dataset.reset_iter_state(nce_noise_bz, noise_ratio = nce_noise_ratio)
            for mm in ['data', 'noise']:
                tmp_inputs = nce_noise_dataset.get_next_batch(mm)
                for kk in range(2):
                    logger.info('check %s %d : %s', mm, kk, self.tokenizer.decode(tmp_inputs['input_ids'][kk]))
            nce_noise_dataset.reset_iter_state(nce_noise_bz, noise_ratio = nce_noise_ratio)

        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss, logging_loss = 0.0, 0.0
        #nce_loss, logging_nce_loss = 0.0, 0.0
        stat_d = {'energy_data_mean': [], 'energy_noise_mean': [], 'energy_data_max': [], 'energy_noise_min': [], 'energy_disc_acc': [], 'nce_loss': [], 'ce_grad': [], 'nce_grad': []}
        eval_res_lis = []
        eval_res_savedir = self.args.output_dir + '/eval_res_save'
        command = 'mkdir -p ' + eval_res_savedir
        os.system(command)
        model.zero_grad()
        
        def full_evaluate():
            eval_res = {'global_step': self.global_step, 'epoch': epoch, 'lr': scheduler.get_lr()[0]}
            res = self.evaluate(input_transform = input_transform); eval_res.update(res)
            ece_res = self.eval_calibration(self.eval_dataset, verbose = False); 
            eval_res['ece'] = ece_res['ece']
            torch.cuda.empty_cache()
            if train_mode == 'nce_noise':
                logger.info('during evaluate, resetting nce_noise_eval_dataset...')
                nce_eval_res = self.nce_evaluate(nce_noise_eval_dataset)
                eval_res.update(nce_eval_res)
            """
            if checklist_suite is not None:
                cres = checklist_utils.run_checklist_suite(model, self.tokenizer, self.eval_dataset, all_args, given_suite = checklist_suite, verbose = False)
                eval_res['checklist_AVG'] = cres['AVG']
            """
            eval_res_lis.append(eval_res)
            logger.info('==eval_res: %s', str(eval_res))
            model.zero_grad()
            save_fn = eval_res_savedir + '/eval_res_step{}.save'.format(self.global_step)
            torch.save(eval_res_lis, save_fn)
            logger.info('eval_res saved to %s', save_fn)

        my_args = self.my_args
        num_labels = len(self.train_dataset.label_list)
        logger.info('num_labels: %d', num_labels)
        p_emps = None
        if my_args.pcal_train == True:
            bins = setup_bins(my_args.pcal_bin_size)
            self.bins = bins
            p_emps = np.zeros((len(bins)-1, num_labels))
        
        #train_iterator = trange(
        #    epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        #)
        #for epoch in train_iterator:
        for epoch in range(epochs_trained, int(num_train_epochs)):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            
            """
            if is_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_master())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_master())
            """
            if train_mode == 'nce_noise':
                nce_noise_dataset.report_iter_state()
            #for step, inputs in enumerate(epoch_iterator):
            if my_args.pcal_train == True:
                pcal_update_freq = len(train_dataloader) // my_args.pcal_num_updates
                logger.info('pcal_update_freq: %d', pcal_update_freq)

            for step, inputs in enumerate(train_dataloader):
        
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                if input_transform is not None:
                    inputs = input_transform(inputs)
                tr_loss += self._training_step(model, inputs, optimizer, p_emps = p_emps)
                
                ce_grad, ce_gradbook = 0, {}
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        ce_grad += p.grad.view(-1).norm() ** 2
                        ce_gradbook[n] = copy.deepcopy(p.grad)
                #logger.info('ce_grad %f', math.sqrt(ce_grad))
                stat_d['ce_grad'].append(math.sqrt(ce_grad))

                if train_mode == 'nce_noise':
                    stat_d_tmp = self.nce_forward(model, nce_noise_dataset, do_train = True)
                    nce_grad = 0
                    for n, p in model.named_parameters():
                        if p.grad is not None and n in ce_gradbook:
                            nce_grad += (p.grad - ce_gradbook[n]).view(-1).norm() ** 2
                    #logger.info('nce_grad %f', math.sqrt(nce_grad))
                    stat_d['nce_grad'].append(math.sqrt(nce_grad))
                    #tr_nce_loss += np.mean(nce_loss_tmp)
                    for idx in stat_d:
                        if not 'grad' in idx:
                            stat_d[idx].append(np.mean(stat_d_tmp[idx]))
                


                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    #if is_tpu_available():
                    #    xm.optimizer_step(optimizer)
                    #else:
                    optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(train_dataloader) #len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss = tr_loss #logging_loss, logging_nce_loss = tr_loss, tr_nce_loss
                        if train_mode == 'nce_noise':
                            for idx in stat_d:
                                logs[idx] = np.mean(stat_d[idx][-self.args.logging_steps:])
                        #self._log(logs)
                        logger.info('epoch %s step %d %s:', str(self.epoch), self.global_step, str(logs))

                    if self.args.evaluate_during_training and (self.my_args.eval_steps > 0) and (self.global_step > 0) and (self.global_step % self.my_args.eval_steps == 0 or self.global_step == 100 or self.global_step == 200):
                        full_evaluate()
                        logger.info('outputdir is %s', self.args.output_dir)

                    #Add postcal
                    ##Update p_emps
                    if (epoch >= my_args.pcalloss_start_epochs and my_args.pcal_train and ((step+1) % pcal_update_freq == 0)) or ((step+1) == len(train_dataloader) and my_args.pcal_train):
                        freq_p_hats = np.zeros((len(bins)-1, num_labels))
                        freq_p_trues= np.zeros((len(bins)-1, num_labels))
                        acc_p_hats = np.zeros((len(bins)-1, num_labels))

                        print("Update p_emp at epoch %d, steps %d" %(epoch, step+1))
                        cal_dataloader = self.get_train_dataloader()
                        model.eval()
                        for cal_step, cal_batch in enumerate(cal_dataloader):
                            #if cal_step > 400:
                            #    print('DEBUG! stop at cal_step 400')
                            #    break
                            for t in cal_batch: cal_batch[t] = cal_batch[t].cuda()
                            outputs = model(**cal_batch)
                            logits = outputs[1]
                            labels = cal_batch['labels']
                            #if labels.size(0) < logits.size(0):
                            #    breakpoint()

                            p_hats = torch.nn.functional.softmax(logits,dim=1).cpu().detach().numpy()
                            p_binums = np.digitize(p_hats,bins) -1
                            p_trues = np.eye(num_labels)[labels.cpu()]

                            #try:
                            if 1 == 1:
                                for sample in np.dstack((p_binums,p_trues,p_hats)):
                                    for col_idx, subsample in enumerate(sample):
                                        row_idx = int(subsample[0])
                                        indiv_true = subsample[1]
                                        indiv_p_hat = subsample[2]
                                        try:
                                            freq_p_hats[row_idx][col_idx] +=1
                                            freq_p_trues[row_idx][col_idx] +=indiv_true
                                            acc_p_hats[row_idx][col_idx] +=indiv_p_hat
                                        except IndexError:
                                            freq_p_hats[row_idx-1][col_idx] +=1
                                            freq_p_trues[row_idx-1][col_idx] +=indiv_true
                                            acc_p_hats[row_idx-1][col_idx] += indiv_p_hats
                            #except ValueError:
                            #    print('ValueError!!!')
                            #    continue
                            #    #import pdb; pdb.set_trace();

                        p_emps = freq_p_trues/freq_p_hats
                        p_emps[np.isnan(p_emps)]=0

                        #ECE
                        p_confs = acc_p_hats /freq_p_hats
                        p_confs[np.isnan(p_confs)] = 0
                        #train_ece = np.sum(freq_p_hats*(abs(p_confs-p_emps))) / np.sum(freq_p_hats[:,0])
                        train_ece = np.sum(freq_p_hats*(abs(p_confs-p_emps))) / np.sum(freq_p_hats)
                        logger.info('step: %d postcal train_ece: %f', step, train_ece)

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert model.module is self.model
                        else:
                            assert model is self.model
                        # Save model checkpoint
                        output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")

                        self.save_model(output_dir)

                        if self.is_world_master():
                            self._rotate_checkpoints()

                        #if is_tpu_available():
                        #    xm.rendezvous("saving_optimizer_states")
                        #    xm.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        #    xm.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        if self.is_world_master():
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    #epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                #train_iterator.close()
                break
            if self.args.tpu_metrics_debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())
            comm = 'cp ' + self.my_args.log_fn + ' ' + '{}_ep{}.txt'.format(self.my_args.log_fn[:-4], epoch)
            logger.info('copying log_fn: %s', comm)
            os.system(comm)

        logger.info('===END of training, doing full_evaluate')
        full_evaluate()

        if self.tb_writer:
            self.tb_writer.close()

        if train_mode == 'nce_noise':
            logger.info('===nce_noise_dataset FINAL iter state report===')
            nce_noise_dataset.report_iter_state()

        if return_d is not None: 
            return_d['eval_res_lis'] = eval_res_lis
        
        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step)

    def _log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.tb_writer:
            for k, v in logs.items():
                self.tb_writer.add_scalar(k, v, self.global_step)
            self.tb_writer.flush()
        if is_wandb_available():
            wandb.log(logs, step=self.global_step)
        output = json.dumps({**logs, **{"step": self.global_step}})
        if iterator is not None:
            iterator.write(output)
        else:
            logger.info(str(output))

    def _nce_training_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()


    def _training_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer, p_emps = None
    ) -> float:
        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)
        if p_emps is not None:
            inputs['empprob_ref'] = p_emps
            inputs['my_args'] = self.my_args
            inputs['bins'] = self.bins

        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph = True)
        else:
            loss.backward(retain_graph = True)

        return loss.item()

    def is_local_master(self) -> bool:
        #if is_tpu_available():
        #    return xm.is_master_ordinal(local=True)
        #else:
        return self.args.local_rank in [-1, 0]

    def is_world_master(self) -> bool:
        """
        This will be True only in one process, even in distributed mode,
        even when training on multiple machines.
        """
        #if is_tpu_available():
        #    return xm.is_master_ordinal(local=False)
        #else:
        return self.args.local_rank == -1 or torch.distributed.get_rank() == 0

    def save_model(self, output_dir: Optional[str] = None):
        """
        Saving best-practices: if you use default names for the model,
        you can reload it using from_pretrained().

        Will only save from the world_master process (unless in TPUs).
        """

        #if is_tpu_available():
        #    self._save_tpu(output_dir)
        if self.is_world_master():
            self._save(output_dir)

    def _save_tpu(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        logger.info("Saving model checkpoint to %s", output_dir)

        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")

        xm.rendezvous("saving_checkpoint")
        self.model.save_pretrained(output_dir)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self.model.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.args.output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    def evaluate(
        self, eval_dataset: Optional[Dataset] = None, prediction_loss_only: Optional[bool] = None, input_transform = None
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent.

        Args:
            eval_dataset: (Optional) Pass a dataset if you wish to override
            the one on the instance.
        Returns:
            A dict containing:
                - the eval loss
                - the potential metrics computed from the predictions
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self._prediction_loop(eval_dataloader, description="Evaluation", input_transform = input_transform)

        self._log(output.metrics)

        if self.args.tpu_metrics_debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output.metrics

    def predict(self, test_dataset: Dataset) -> PredictionOutput:
        """
        Run prediction and return predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in evaluate().
        """
        test_dataloader = self.get_test_dataloader(test_dataset)

        return self._prediction_loop(test_dataloader, description="Prediction")

    def _prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None, input_transform = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        #logger.info("***** Running %s *****", description)
        logger.info("** %s **  Num examples = %d", description, self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        #if is_tpu_available():
        #    dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)
            if input_transform is not None:
                inputs = input_transform(inputs)
                 
            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            if not prediction_loss_only:
                if preds is None:
                    preds = logits.detach()
                else:
                    preds = torch.cat((preds, logits.detach()), dim=0)
                if inputs.get("labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["labels"].detach()
                    else:
                        label_ids = torch.cat((label_ids, inputs["labels"].detach()), dim=0)

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = self.distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = self.distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
        """
        elif is_tpu_available():
            # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
            if preds is not None:
                preds = xm.mesh_reduce("eval_preds", preds, torch.cat)
            if label_ids is not None:
                label_ids = xm.mesh_reduce("eval_label_ids", label_ids, torch.cat)
        """

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}

        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def distributed_concat(self, tensor: torch.Tensor, num_total_examples: int) -> torch.Tensor:
        assert self.args.local_rank != -1

        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)

        concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        output = concat[:num_total_examples]
        return output
    

