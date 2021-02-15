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
"""PyTorch RoBERTa model. """


import logging
import warnings

import torch
import torch.nn as nn
#from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import CrossEntropyLoss, MSELoss, KLDivLoss

from transformers.configuration_roberta import RobertaConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable
from transformers.modeling_bert import BertEmbeddings, BertLayerNorm, BertModel, BertPreTrainedModel, gelu
#from transformers.modeling_utils import create_position_ids_from_input_ids
from transformers.modeling_roberta import RobertaModel, RobertaClassificationHead
import numpy as np
import copy, sys

logger = logging.getLogger(__name__)

class MyScalarHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class MyRobertaForNCESequenceClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.myclassifier = RobertaClassificationHead(config)
        self.myscalar = MyScalarHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        special_mode = 'normal',
        nce_mode = None, #'normal',
        nce_feed_type = None, #'data', #data or noise
        nce_noise_ratio = None,
        return_d = None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import RobertaTokenizer, RobertaForSequenceClassification
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

        """
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        logits = self.myclassifier(sequence_output)
        outputs = (logits,) + outputs[2:]

        loss = None
        if labels is not None:
            #print('debug special_mode:', special_mode)
            if special_mode == 'normal':
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            if special_mode == 'nce_noise':        
                if nce_mode == 'normal':
                    nce_logits = self.myscalar(sequence_output)
                    assert(nce_logits.size(1) == 1)
                elif nce_mode == 'negnormal':
                    nce_logits = - self.myscalar(sequence_output)
                    assert(nce_logits.size(1) == 1)
                elif nce_mode == 'hidden':
                    assert(logits.size(1) > 1)
                    nce_logits = - torch.logsumexp(logits, dim = -1, keepdim = True)
                elif nce_mode == 'neghidden':
                    assert(logits.size(1) > 1)
                    nce_logits = torch.logsumexp(logits, dim = -1, keepdim = True)
                elif nce_mode == 'labeled':
                    assert(logits.size(1) > 1)
                    nce_logits = - logits.gather(1, labels.view(-1, 1))
                elif nce_mode == 'selflabeled':
                    assert(logits.size(1) > 1)
                    self_labels = torch.argmax(logits, dim = 1)
                    nce_logits = - logits.gather(1, self_labels.view(-1, 1))
                elif nce_mode == 'noise_selflabeled':
                    assert(logits.size(1) > 1)
                    #breakpoint()
                    if nce_feed_type == 'noise':
                        self_labels = torch.argmax(logits, dim = 1)
                        nce_logits = - logits.gather(1, self_labels.view(-1, 1))
                    else:
                        nce_logits = - logits.gather(1, labels.view(-1, 1))
                elif nce_mode == 'negselflabeled':
                    assert(logits.size(1) > 1)
                    self_labels = torch.argmax(logits, dim = 1)
                    nce_logits = logits.gather(1, self_labels.view(-1, 1))
                elif nce_mode == 'selfsamplelabeled':
                    assert(logits.size(1) > 1)
                    l_distro = torch.softmax(logits, dim = 1)
                    #self_labels = torch.argmax(logits, dim = 1)
                    self_labels = torch.multinomial(l_distro, 1)
                    nce_logits = - logits.gather(1, self_labels.view(-1, 1))
                else:
                    logger.info('something wrong with nce_mode! %s', nce_mode)
                    sys.exit(1)
                scal = torch.ones(nce_logits.size()).cuda()
                #logger.info('debug nce_mode: %s nce_feed_type: %s nce_noise_ratio: %d', nce_mode, nce_feed_type, nce_noise_ratio)
                #originally, label == 0 means noise, label == 1 means data
                if nce_feed_type == 'noise':
                    loss_batch = torch.log(nce_noise_ratio + torch.exp(nce_logits * -1)) #careful about bug (negative)! minize+minus(log is inversed) is enough!
                elif nce_feed_type == 'data':
                    loss_batch = torch.log(1 + nce_noise_ratio * torch.exp(nce_logits * 1)) 
                else:
                    logger.info('ERROR nce_feed_type: %s', nce_feed_type)
                    sys.exit(1)
                #print(nce_logits)
                #TODO: return_d
                if nce_mode in ['selflabeled', 'selfsamplelabeled']:
                    labels = self_labels
                return_d['energy_data_mean'] = None
                return_d['energy_noise_mean'] = None
                if nce_feed_type == 'data':
                    energy_data_mean = nce_logits.mean().item()
                    return_d['energy_data_mean'] = energy_data_mean
                    return_d['energy_data_max'] = nce_logits.max().item()
                if nce_feed_type == 'noise':
                    energy_noise_mean = nce_logits.mean().item()
                    return_d['energy_noise_mean'] = energy_noise_mean
                    return_d['energy_noise_min'] = nce_logits.min().item()
                return_d['nce_logits'] = nce_logits
                loss = loss_batch.mean()

        if loss is not None:
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class MyRobertaForSequenceClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        empprob_ref=None,
        my_args = None,
        bins = None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import RobertaTokenizer, RobertaForSequenceClassification
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

        """
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if empprob_ref is not None and np.sum(empprob_ref) > 0:
                loss_mle, loss_cal = torch.zeros(1), torch.zeros(1)
                pred = torch.nn.functional.softmax(logits, dim=1)
                empprob = np.zeros(pred.shape)
                pred_to_empprob_map = np.digitize(pred.cpu().detach().numpy(), bins) - 1
                for sample, empmaps in enumerate(pred_to_empprob_map):
                    for clas, empprob_idx in enumerate(empmaps):
                        try:
                            empprob[sample][clas] = empprob_ref[empprob_idx][clas]
                        except IndexError:
                            empprob[sample][clas] = empprob_ref[empprob_idx - 1][clas]

                empprob = torch.from_numpy(empprob).float().cuda()
                if "MSE" in my_args.pcalloss_type:
                    loss_cal_fct = MSELoss()
                elif my_args.pcalloss_type == "KL":
                    loss_cal_fct = KLDivLoss(reduction="batchmean")
                    pred = torch.log(pred)

                # Update original loss containing calibration loss
                loss_cal = loss_cal_fct(pred.view(-1), empprob.view(-1))
                if my_args.pcalloss_type == "RMSE":
                    loss_cal = torch.sqrt(loss_cal)

                loss_cal = my_args.pcalloss_lambda * loss_cal
                loss += loss_cal

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


