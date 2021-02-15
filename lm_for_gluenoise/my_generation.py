import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional
#from my_lm_trainer import MyTrainer 
#from nce_lm_trainer import MyNCELMTrainer 

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
    set_seed,
)
import torch
import torch.nn.functional as F
#from transformers import GlueDataTrainingArguments as DataTrainingArguments
#from transformers import GlueDataset
from my_glue_dataset import MyGlueDataset, GlueDataTrainingArguments
from my_data_collator import MyDataCollatorForLanguageModeling
import scheduler
import numpy as np

logger = logging.getLogger(__name__)

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_p==None and top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    elif top_k==None and top_p <= 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

debug_lis = []

def generate_no_beam_search(
    model,
    tokenizer,
    input_ids,
    #cur_len,
    max_length,
    #do_sample,
    #temperature,
    schedule,
    #repetition_penalty,
    #bos_token_id,
    #pad_token_id,
    #eos_token_id,
    #decoder_start_token_id,
    #batch_size,
    #attention_mask,
    #dry_run,
    encoder_outputs = None,
    my_args = None,
):
    """ Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
    """
    batch_size = input_ids.size(0)
    eos_token_id, pad_token_id = tokenizer.eos_token_id, tokenizer.pad_token_id
    temperature = 1.0
    # length of generated sentences / unfinished sentences
    unfinished_sents = input_ids.new(batch_size).fill_(1)
    sent_lengths = input_ids.new(batch_size).fill_(max_length)

    past = encoder_outputs  # defined for encoder-decoder models, None for decoder-only models
    all_model_logits = [] # initialize distribution over probabilities all_transformed_logits = [] # initialize transformed distribution over probabilities

    cur_len = input_ids.size(1)
    attention_mask = torch.ones(input_ids.size()).cuda()

    while cur_len < max_length:
        step = schedule.step()
        if isinstance(schedule, scheduler.JointScheduler):
            top_p = step['top_p']
            top_k = step['top_k']
        elif schedule.is_top_p:
            top_p = step
            top_k = None
        else:
            top_k = step
            top_p = None

        model_inputs = model.prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask, use_cache=True)
        assert 'attention_mask' not in model_inputs
        # print(attention_mask)
        # print(model_inputs.keys())
        outputs = model(**model_inputs)
        all_model_logits.append(outputs[0].cpu().detach())
        next_token_logits = outputs[0][:, -1, :]

        # if model has past, then set the past variable to speed up decoding
        if model._use_cache(outputs, True):
            past = outputs[1]

        # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
        #if repetition_penalty != 1.0:
        #    model.enforce_repetition_penalty_(next_token_logits, batch_size, 1, input_ids, repetition_penalty)

        if True:
            if isinstance(schedule, scheduler.NoisedTemperatureScheduler):
                #transform it before temperature
                next_token_logits = schedule.transform(next_token_logits)
            
            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            if cur_len == my_args.noiselm_partial2full_prefixlen + 1 and my_args.noiselm_mode in ['labeledP2F', 'labeledH2F']:
                #logger.info('forced temperature %f', my_args.noiselm_labeledP2F_labelsample_temperature)
                next_token_logits = next_token_logits / my_args.noiselm_labeledP2F_labelsample_temperature
                next_token_logits[:, :tokenizer.encode('<LABEL0>')[0]] = -30
                next_token_logits[:, (tokenizer.encode('<LABEL0>')[0] + len(my_args.label_list)):] = -30

            # Check if we are using a joint scheduler
            if isinstance(schedule, scheduler.JointScheduler):
                # do a min(top_p, top_k) 
                next_token_logits = combined_filtering(next_token_logits, top_k=top_k, top_p=top_p) 
            else:
                # Top-p/top-k/Simplex filtering
                least_value = torch.min(next_token_logits) - 1000
                if isinstance(schedule, scheduler.UniformSimplexScheduler) or isinstance(schedule, scheduler.SortedSimplexScheduler):
                    next_token_logits = schedule.transform(next_token_logits, temperature = temperature, least_value=least_value)
                elif isinstance(schedule, scheduler.TargetEntropyScheduler) or isinstance(schedule, scheduler.MaxEntropyScheduler) or isinstance(schedule, scheduler.RandomSpaceTopkScheduler):
                    assert(temperature == 1.0)
                    next_token_logits = schedule.transform(next_token_logits)
                elif isinstance(schedule, scheduler.SortedNoisedFixedScheduler):
                    next_token_logits = schedule.transform(next_token_logits)
                elif isinstance(schedule, scheduler.NoisedTemperatureScheduler):
                    pass
                    #we did the transform before the temperaturetuning
                    #next_token_logits = schedule.transform(next_token_logits)
                else:
                    assert(schedule, scheduler.FixedScheduler)
                    next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p, filter_value=least_value)

            #all_transformed_logits.append(next_token_logits.unsqueeze(1).cpu().detach())
            probs = F.softmax(next_token_logits, dim=-1)
            #if dry_run and (schedule.is_top_p or isinstance(schedule, scheduler.JointScheduler)):
            #    print("Number of tokens sampled from:", (probs > 0).sum(dim=1))
            #    print()
            
            """
            probs_ori = copy.deepcopy(probs)
            if isinstance(schedule, scheduler.UniformSimplexScheduler):
                for kk in range(probs.size(0)):
                    co = 0
                    while torch.sum(probs[kk]).item() > 1 - 1e-03:
                        probs[kk] = probs[kk] * (1 - 1e-3)
                        #print(co, end = ' ')
                        co = co + 1
                        #sys.stdout.flush()
            
            fn = 'tmp/logit_debug.save'
            torch.save({'probs': probs, 'probs_ori': probs_ori, 'logits': next_token_logits, 'time': str(datetime.datetime.now())}, fn)
            print(fn, datetime.datetime.now())
            time.sleep(0.1)
            """
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            #for debug
            sample_logprobs = torch.log(probs.gather(1, next_token.view(-1, 1)))
            if len(debug_lis) < 102 and cur_len < 30:
                #print(cur_len, torch.mean(sample_logprobs))
                debug_lis.append(torch.mean(sample_logprobs).item())
                if len(debug_lis) % 10 == 0:
                    print('sample_logprob debug_lis: len:{} mean:{}'.format(len(debug_lis), np.mean(debug_lis)))
        else:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1)

        # update generations and finished sentences
        if eos_token_id is not None:
            # pad finished sentences if eos_token_id exist
            tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
        else:
            tokens_to_add = next_token

        input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

        if eos_token_id is not None:
            eos_in_sents = tokens_to_add == eos_token_id
            # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
            is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
            sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len + 1)
            # unfinished_sents is set to zero if eos in sentence
            unfinished_sents.mul_((~eos_in_sents).long())

        # stop when there is a </s> in each sentence, or if we exceed the maximul length
        if unfinished_sents.max() == 0:
            break

        # extend attention_mask for new generated input if only decoder
        if model.config.is_encoder_decoder is False:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        cur_len = cur_len + 1

    """
    # if there are different sentences lengths in the batch, some batches have to be padded
    if sent_lengths.min().item() != sent_lengths.max().item():
        assert pad_token_id is not None, "`Pad_token_id` has to be defined if batches have different lengths"
        # finished sents are filled with pad_token
        decoded = input_ids.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)
    else:
        decoded = input_ids

    for hypo_idx, hypo in enumerate(input_ids):
        decoded[hypo_idx, : sent_lengths[hypo_idx]] = hypo[: sent_lengths[hypo_idx]]

    all_model_logits = torch.cat(all_model_logits, dim=1)
    all_transformed_logits = torch.cat(all_transformed_logits, dim=1)
    """
    return input_ids


