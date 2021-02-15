import checklist
import spacy
import itertools

import checklist.editor
#import checklist.text_generation
from checklist.test_types import MFT, INV, DIR
from checklist.expect import Expect
from checklist.test_suite import TestSuite
import numpy as np
import spacy
from checklist.perturb import Perturb
import collections
from collections import defaultdict, OrderedDict

import dataclasses
import logging
import os, math, re
import sys, copy, random
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
#from my_trainer import MyTrainer 
from my_glue_dataset import MyGlueDataset
from my_modeling_roberta import MyRobertaForSequenceClassification, MyRobertaForNCESequenceClassification
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from my_utils import setLogger 

logger = logging.getLogger()
editor = checklist.editor.Editor()   
nlp = spacy.load('en_core_web_sm')

def wrap_apply_to_each(fn, both=False, *args, **kwargs):
    def new_fn(qs, *args, **kwargs):
        q1, q2 = qs
        ret = []
        fnq1 = fn(q1, *args, **kwargs)
        fnq2 = fn(q2, *args, **kwargs)
        if type(fnq1) != list:
            fnq1 = [fnq1]
        if type(fnq2) != list:
            fnq2 = [fnq2]
        ret.extend([(x, str(q2)) for x in fnq1])
        ret.extend([(str(q1), x) for x in fnq2])
        if both:
            ret.extend([(x, x2) for x, x2 in itertools.product(fnq1, fnq2)])
        return [x for x in ret if x[0] and x[1]]
    return new_fn

def wrap_apply_to_both(fn, *args, **kwargs):
    def new_fn(qs, *args, **kwargs):
        q1, q2 = qs
        ret = []
        fnq1 = fn(q1, *args, **kwargs)
        fnq2 = fn(q2, *args, **kwargs)
        if type(fnq1) != list:
            fnq1 = [fnq1]
        if type(fnq2) != list:
            fnq2 = [fnq2]
        ret.extend([(x, x2) for x, x2 in itertools.product(fnq1, fnq2)])
        return [x for x in ret if x[0] and x[1]]
    return new_fn

def my_summary(self, types=None, capabilities=None, verbose = True, **kwargs):
    """Print stats and example failures for each test.
    See summary in abstract_test.py
    Parameters
    ----------
    types : list(string)
        If not None, will only show tests of these test types.
        Options are MFT, INV, and DIR
    capabilities : list(string)
        If not None, will only show tests with these capabilities.
    **kwargs : type
        Will be passed as arguments to each test.summary()
    """
    vals = collections.defaultdict(lambda: 100, {'MFT': 0, 'INV': 1, 'DIR': 2})
    tests = self.tests.keys()
    capability_order = ['Vocabulary', 'Taxonomy', 'Robustness', 'NER',  'Fairness', 'Temporal', 'Negation', 'Coref', 'SRL', 'Logic']
    cap_order = lambda x:capability_order.index(x) if x in capability_order else 100
    caps = sorted(set([x['capability'] for x in self.info.values()]), key=cap_order)
    res_failrate = {}
    for capability in caps:
        if capabilities is not None and capability not in capabilities:
            continue
        if verbose:
            print(capability)
            print()
        tests = [x for x in self.tests if self.info[x]['capability'] == capability]
        for n in tests:
            if types is not None and self.info[n]['type'] not in types:
                continue
            if verbose:
                print(n)
            if 'format_example_fn' not in kwargs:
                kwargs['format_example_fn'] = self.info[n].get('format_example_fn', self.format_example_fn)
            if 'print_fn' not in kwargs:
                kwargs['print_fn'] = self.info[n].get('print_fn', self.print_fn)
            if verbose: 
                self.tests[n].summary(**kwargs)
            ss = self.tests[n].get_stats()
            res_failrate[capability.upper()[:3] + '_' + n.replace(' ', '_')] = ss.fail_rate / 100.0
            if verbose:
                print()
                print()
        if verbose:
            print()
            print()
    ll = []
    for idx in res_failrate:
        ll.append(res_failrate[idx])
    res_failrate['AVG'] = np.mean(ll)
    return res_failrate

def do_checklist_QQP(model, tokenizer, glue_dataset, all_args, only_construct_suite = False, given_suite = None, verbose = True):
    logger.info('do_checklist_QQP')
    model_args, data_args, training_args, my_args = all_args 

    if only_construct_suite == True or given_suite is None:
        qs = []
        labels = []
        all_questions = set()
        for x in open(data_args.data_dir + '/dev.tsv').readlines()[1:]:
            try:
                q1, q2, label = x.strip().split('\t')[3:]
            except:
                print('warning: discarded', x.strip())
                continue
            all_questions.add(q1)
            all_questions.add(q2)
            qs.append((q1, q2))
            labels.append(label)
            #if len(labels) > 1000: 
            #    logger.info('DEBUG break')
            #    break 
        labels = np.array(labels).astype(int)

        all_questions = list(all_questions)
        parsed_questions = list(nlp.pipe(all_questions))
        spacy_map = dict([(x, y) for x, y in zip(all_questions, parsed_questions)])

        parsed_qs = [(spacy_map[q[0]], spacy_map[q[1]]) for q in qs]
        
        logger.info('constructing test suite')
        suite = TestSuite()
        
        t = Perturb.perturb(qs, wrap_apply_to_each(Perturb.add_typos), nsamples= 2000)
        test = INV(t.data, name='add one typo', capability='Robustness', description='')
        # test.run(new_pp)
        # test.summary(3)
        suite.add(test, overwrite=True)    
       
        import itertools
        def me_to_you(text):
            t = re.sub(r'\bI\b', 'you', text)
            t = re.sub(r'\bmy\b', 'your', t)
            return re.sub(r'\bmine\b', 'yours', t)
        def paraphrases(text):
            ts = ['How do I ', 'How can I ', 'What is a good way to ', 'How should I ']
            templates1 = ['How do I {x}?', 'How can I {x}?', 'What is a good way to {x}?', 'If I want to {x}, what should I do?',
                        'In order to {x}, what should I do?']
            ts2 = ['Can you ', 'Can I ']#, 'Do I']
            ts3 = ['Do I ']
            templates2 = ['Can you {x}?', 'Can I {x}?', 'Do you think I can {x}?', 'Do you think you can {x}?',]
            templates3 = ['Do I {x}?', 'Do you think I {x}?']
            ret = []
            for i, (tsz, templates) in enumerate(zip([ts, ts2, ts3], [templates1, templates2, templates3])):
                for t in tsz:
                    if text.startswith(t):
                        x = text[len(t):].strip('?')
                        ts = editor.template(templates, x=x).data[0]
                        if i <= 1:
                            ts = ts + [me_to_you(x) for x in ts]
                        ret += ts
            return ret
        def paraphrases_product(text):
            pr = paraphrases(text)
            return list(itertools.product(pr, pr))
     
        def paraphrase_each(pair):
            p1 = paraphrases(pair[0])
            p2 = paraphrases(pair[1])
            return list(itertools.product(p1, p2)) 
        
        t = Perturb.perturb(list(all_questions), paraphrases_product, nsamples= 2000, keep_original=False)
        name = '(q, paraphrase(q))'
        desc = 'For questions that start with "How do I X", "How can I X", etc'
        test = DIR(t.data, expect=Expect.eq(1), agg_fn='all', name=name, description=desc, capability='Robustness')
        suite.add(test)
        
        t = Perturb.perturb(qs, paraphrase_each, nsamples= 2000, keep_original=True)
        name = 'Product of paraphrases(q1) * paraphrases(q2)'
        desc = 'For questions that start with "How do I X", "How can I X", etc'
        test = INV(t.data, name=name, description=desc, capability='Robustness')
        # test.run(new_pp)
        # test.summary(n=5)
        suite.add(test)
        
        logger.info('constructing test suite complete')
    else:
        suite = given_suite

    if only_construct_suite:
        return suite
  
    #from pattern.en import sentiment
    def predict_proba(inputs):
        features = glue_dataset.convert_checklist_input(inputs)
        if verbose: 
            print('len(inputs)', len(inputs))
            print('debug inputs[0] after conversion', tokenizer.decode(features[0].input_ids))
        model.eval()
        idx_now, bz, probs = 0, 8, []
        while idx_now < len(features):
            input_ids = torch.LongTensor([f.input_ids for f in features[idx_now:idx_now + bz]]).cuda()
            attention_mask = torch.LongTensor([f.attention_mask for f in features[idx_now:idx_now + bz]]).cuda()
            idx_now += bz
            outputs = model(input_ids = input_ids, attention_mask = attention_mask)
            logits = outputs[0]
            prob = torch.softmax(logits, dim = -1).cpu().detach()
            probs.append(prob)

        """
        p1 = np.array([(sentiment(x)[0] + 1)/2. for x in inputs]).reshape(-1, 1)
        p1 = np.random.uniform(size=p1.shape)
        p0 = 1- p1
        pp = np.hstack((p0, p1))
        """
        pp = torch.cat(probs, dim = 0).numpy()
        return pp
    
    from checklist.pred_wrapper import PredictorWrapper
    wrapped_pp = PredictorWrapper.wrap_softmax(predict_proba)
    
    suite.run(wrapped_pp, overwrite=True)
    #suite.summary()
    res = my_summary(suite, verbose = verbose)

    logger.info('do_checklist_QQP complete')
    return res

def do_checklist_QNLI(model, tokenizer, glue_dataset, all_args, only_construct_suite = False, given_suite = None, verbose = True):
    logger.info('do_checklist_QNLI')
    model_args, data_args, training_args, my_args = all_args 
    if only_construct_suite == True or given_suite is None:
        logger.info('label_list: %s', str(glue_dataset.label_list))
        qs = []
        labels = []
        all_qs = set()
        random_answers = set()
        for x in open(data_args.data_dir + '/dev.tsv').readlines()[1:]:
            try:
                q1, q2, label = x.strip().split('\t')[1:]
            except:
                print('warning: discarded', x.strip())
                continue
            all_qs.add(q1); all_qs.add(q2)
            random_answers.add(q2)
            qs.append((q1, q2))
            assert(label in ['entailment', 'not_entailment'])
            labels.append(0 if label == 'entailment' else 1)
            #if len(labels) > 1000: 
            #    logger.info('DEBUG break')
            #    break 
        labels = np.array(labels).astype(int)

        all_qs = list(all_qs)
        random_answers = list(random_answers)
        parsed_qs = list(nlp.pipe(all_qs))
        spacy_map = dict([(x, y) for x, y in zip(all_qs, parsed_qs)])
        processed_qs = [(spacy_map[q[0]], spacy_map[q[1]]) for q in qs]
        
        logger.info('constructing test suite')
        suite = TestSuite()

        def question_typo(x):
            return (Perturb.add_typos(x[0]), Perturb.add_typos(x[1]))
        t = Perturb.perturb(qs, question_typo, nsamples=500)
        test = INV(t.data, name='both typo', capability='Robustness', description='')
        # test.run(new_pp)
        # test.summary(3)
        suite.add(test, overwrite=True)    
        
        def add_random_answer(x):
            random_s = np.random.choice(random_answers)
            while random_s in x[1]:
                random_s = np.random.choice(random_answers)
            return (x[0], x[1].strip() + ' ' + random_s)

        from checklist.expect import Expect
        monotonic_decreasing = Expect.monotonic(label=0, increasing=False, tolerance=0.1)
        
        t = Perturb.perturb(qs, add_random_answer, nsamples = 1000)
        test = DIR(**t, expect=monotonic_decreasing)
        suite.add(test, name='add random answer', capability='Robustness')
      
        def change_thing(change_fn):
            def change_both(cq, **kwargs):
                context, question = cq
                a = change_fn(context, meta=True)
                if not a:
                    return None
                changed, meta = a
                ret = []
                for c, m in zip(changed, meta):
                    new_q = re.sub(r'\b%s\b' % re.escape(m[0]), m[1], question.text)
                    ret.append((c, new_q))
                return ret, meta
            return change_both
        t = Perturb.perturb(processed_qs, change_thing(Perturb.change_names), nsamples=1000, meta=True)
        test = INV(**t, name='Change name everywhere', capability='Robustness', description='')
        suite.add(test)

        t = Perturb.perturb(processed_qs, change_thing(Perturb.change_location), nsamples=1000, meta=True)
        test = INV(**t, name='Change location everywhere', capability='Robustness', description='')
        suite.add(test)

        logger.info('constructing test suite complete')
    else:
        suite = given_suite
        
    if only_construct_suite == True:
        return suite
      
    #from pattern.en import sentiment
    def predict_proba(inputs):
        features = glue_dataset.convert_checklist_input(inputs)
        if verbose == True: 
            print('len(inputs)', len(inputs))
            print('debug inputs[0] after conversion', tokenizer.decode(features[0].input_ids))
        model.eval()
        idx_now, bz, probs = 0, 8, []
        while idx_now < len(features):
            input_ids = torch.LongTensor([f.input_ids for f in features[idx_now:idx_now + bz]]).cuda()
            attention_mask = torch.LongTensor([f.attention_mask for f in features[idx_now:idx_now + bz]]).cuda()
            idx_now += bz
            outputs = model(input_ids = input_ids, attention_mask = attention_mask)
            logits = outputs[0]
            prob = torch.softmax(logits, dim = -1).cpu().detach()
            probs.append(prob)

        """
        p1 = np.array([(sentiment(x)[0] + 1)/2. for x in inputs]).reshape(-1, 1)
        p1 = np.random.uniform(size=p1.shape)
        p0 = 1- p1
        pp = np.hstack((p0, p1))
        """
        pp = torch.cat(probs, dim = 0).numpy()
        return pp
    
    from checklist.pred_wrapper import PredictorWrapper
    wrapped_pp = PredictorWrapper.wrap_softmax(predict_proba)
    
    suite.run(wrapped_pp, overwrite=True)
    #suite.summary()
    res = my_summary(suite, verbose = verbose)

    logger.info('do_checklist_QNLI complete')
    return res

def do_checklist_SST2(model, tokenizer, glue_dataset, all_args, only_construct_suite = False, given_suite = None, verbose = True):
    logger.info('do_checklist_SST2')
    model_args, data_args, training_args, my_args = all_args 
    
    if only_construct_suite == True or given_suite == None:
        logger.info('label_list: %s', str(glue_dataset.label_list))
        qs = []
        labels = []
        all_qs = set()
        random_qs = set()
        for x in open(data_args.data_dir + '/dev.tsv').readlines()[1:]:
            try:
                q1, label = x.strip().split('\t')
            except:
                print('warning: discarded', x.strip())
                continue
            all_qs.add(q1); 
            random_qs.add(q1)
            qs.append((q1,))
            assert(label in ['0', '1'])
            labels.append(int(label))
            #if len(labels) > 1000: 
            #    logger.info('DEBUG break')
            #    break 
        labels = np.array(labels).astype(int)

        all_qs = list(all_qs)
        random_qs = list(random_qs)
        parsed_qs = list(nlp.pipe(all_qs))
        spacy_map = dict([(x, y) for x, y in zip(all_qs, parsed_qs)])
        processed_qs = [(spacy_map[q[0]]) for q in qs]
        
        logger.info('constructing test suite')
        suite = TestSuite()

        def question_typo(x):
            return (Perturb.add_typos(x[0]),)
        t = Perturb.perturb(qs, question_typo, nsamples=500)
        test = INV(t.data, name='typo', capability='Robustness', description='')
        # test.run(new_pp)
        # test.summary(3)
        suite.add(test, overwrite=True)    
        
        """ #did not work much
        def word_repeat(x):
            tt = x[0].split()
            k = random.randint(0, len(tt) - 1)
            tt = tt[:k] + [tt[k]] + tt[k:]
            return (' '.join(tt),)
        t = Perturb.perturb(qs, word_repeat, nsamples=500)
        test = INV(t.data, name='word repeat', capability='Robustness', description='')
        # test.run(new_pp)
        # test.summary(3)
        suite.add(test, overwrite=True)    
        """

        pos_sentences = ["It 's hard to describe how much i enjoyed it .", 'I really want to watch it again .', 'I will argue with anyone who hates it .', 'How can anyone resist it .', 'I find it hard to describe how good it is .']
        def add_random_pos(x):
            random_s = np.random.choice(pos_sentences)
            return (x[0] + ' ' + random_s, )
        from checklist.expect import Expect
        l0_monotonic_increasing = Expect.monotonic(label=0, increasing=True, tolerance=0.1)
        l1_monotonic_increasing = Expect.monotonic(label=1, increasing=True, tolerance=0.1)
        t = Perturb.perturb(qs, add_random_pos, nsamples = 1000)
        test = DIR(**t, expect=l1_monotonic_increasing)
        suite.add(test, name='add random positive', capability='Robustness')
        
        editor = checklist.editor.Editor()
        movie_noun = ['movie', 'film', 'shoot', 'experience', 'video']
        editor.add_lexicon('movie_noun', movie_noun)
        pos_adj = ['good', 'great', 'excellent', 'amazing', 'extraordinary', 'beautiful', 'fantastic', 'nice', 'incredible', 'exceptional', 'awesome', 'perfect', 'fun', 'happy', 'adorable', 'brilliant', 'exciting', 'sweet', 'wonderful']
        neg_adj = ['awful', 'bad', 'horrible', 'weird', 'rough', 'lousy', 'unhappy', 'average', 'difficult', 'poor', 'sad', 'frustrating', 'hard', 'lame', 'nasty', 'annoying', 'boring', 'creepy', 'dreadful', 'ridiculous', 'terrible', 'ugly', 'unpleasant']
        neutral_adj = ['American', 'international',  'commercial', 'British', 'private', 'Italian', 'Indian', 'Australian', 'Israeli', ]
        editor.add_lexicon('pos_adj', pos_adj, overwrite=True)
        editor.add_lexicon('neg_adj', neg_adj, overwrite=True )
        editor.add_lexicon('neutral_adj', neutral_adj, overwrite=True)
        pos_verb_present = ['like', 'enjoy', 'appreciate', 'love',  'recommend', 'admire', 'value', 'welcome']
        neg_verb_present = ['hate', 'dislike', 'regret',  'abhor', 'dread', 'despise' ]
        neutral_verb_present = ['see', 'find']
        pos_verb_past = ['liked', 'enjoyed', 'appreciated', 'loved', 'admired', 'valued', 'welcomed']
        neg_verb_past = ['hated', 'disliked', 'regretted',  'abhorred', 'dreaded', 'despised']
        neutral_verb_past = ['saw', 'found']
        editor.add_lexicon('pos_verb_present', pos_verb_present, overwrite=True)
        editor.add_lexicon('neg_verb_present', neg_verb_present, overwrite=True)
        editor.add_lexicon('neutral_verb_present', neutral_verb_present, overwrite=True)
        editor.add_lexicon('pos_verb_past', pos_verb_past, overwrite=True)
        editor.add_lexicon('neg_verb_past', neg_verb_past, overwrite=True)
        editor.add_lexicon('neutral_verb_past', neutral_verb_past, overwrite=True)
        editor.add_lexicon('pos_verb', pos_verb_present+ pos_verb_past, overwrite=True)
        editor.add_lexicon('neg_verb', neg_verb_present + neg_verb_past, overwrite=True)
        editor.add_lexicon('neutral_verb', neutral_verb_present + neutral_verb_past, overwrite=True)
        intens_adj = ['very', 'really', 'absolutely', 'truly', 'extremely', 'quite', 'incredibly', 'amazingly', 'especially', 'exceptionally', 'unbelievably', 'utterly', 'exceedingly', 'rather', 'totally', 'particularly']
        intens_verb = [ 'really', 'absolutely', 'truly', 'extremely',  'especially',  'utterly',  'totally', 'particularly', 'highly', 'definitely', 'certainly', 'genuinely', 'honestly', 'strongly', 'sure', 'sincerely']
        t = editor.template('{it} {movie_noun} {nt} {pos_adj}.', it=['This', 'That', 'The'], nt=['is not', 'isn\'t'], save=True)
        t += editor.template('{it} {benot} {a:pos_adj} {movie_noun}.', it=['It', 'This', 'That'], benot=['is not',  'isn\'t', 'was not', 'wasn\'t'], save=True)
        neg = ['I can\'t say I', 'I don\'t', 'I would never say I', 'I don\'t think I', 'I didn\'t' ]
        t += editor.template('{neg} {pos_verb_present} {the} {movie_noun}.', neg=neg, the=['this', 'that', 'the'], save=True)
        t += editor.template('No one {pos_verb_present}s {the} {movie_noun}.', neg=neg, the=['this', 'that', 'the'], save=True)
        test = MFT(t.data, labels=0, templates=t.templates)
        suite.add(test, 'simple negations: negative', 'Negation', 'Very simple negations of positive statements')


        t = editor.template('{it} {movie_noun} {nt} {neg_adj}.', it=['This', 'That', 'The'], nt=['is not', 'isn\'t'], save=True)
        t += editor.template('{it} {benot} {a:neg_adj} {movie_noun}.', it=['It', 'This', 'That'], benot=['is not',  'isn\'t', 'was not', 'wasn\'t'], save=True)
        neg = ['I can\'t say I', 'I don\'t', 'I would never say I', 'I don\'t think I', 'I didn\'t' ]
        t += editor.template('{neg} {neg_verb_present} {the} {movie_noun}.', neg=neg, the=['this', 'that', 'the'], save=True)
        t += editor.template('No one {neg_verb_present}s {the} {movie_noun}.', neg=neg, the=['this', 'that', 'the'], save=True)
        # expectation: prediction is not 0
        is_not_0 = lambda x, pred, *args: pred != 0
        test = MFT(t.data, Expect.single(is_not_0), templates=t.templates)
        suite.add(test, 'simple negations: not negative', 'Negation', 'Very simple negations of negative statements. Expectation requires prediction to NOT be negative (i.e. neutral or positive)')
        
        t = editor.template('I thought {it} {movie_noun} would be {pos_adj}, but it {neg}.', neg=['was not', 'wasn\'t'], it=['this', 'that', 'the'], nt=['is not', 'isn\'t'], save=True)
        t += editor.template('I thought I would {pos_verb_present} {the} {movie_noun}, but I {neg}.', neg=['did not', 'didn\'t'], the=['this', 'that', 'the'], save=True)
        test = MFT(t.data, labels=0, templates=t.templates)
        suite.add(test, 'simple negations: I thought x was positive, but it was not (should be negative)', 'Negation', '', overwrite=True)
        
        """did not work much
        neg_sentences = ['No one will like it .', 'I will not watch it again .', 'I tell my friends not to watch it .', 'Really a waste of time .']
        def add_random_neg(x):
            random_s = np.random.choice(neg_sentences)
            return (x[0] + ' ' + random_s, )
        from checklist.expect import Expect
        t = Perturb.perturb(qs, add_random_neg, nsamples = 1000)
        test = DIR(**t, expect=l0_monotonic_increasing)
        suite.add(test, name='add random negative', capability='Robustness')
        """
     
        """
        def change_thing(change_fn):
            def change_both(cq, **kwargs):
                context, question = cq
                a = change_fn(context, meta=True)
                if not a:
                    return None
                changed, meta = a
                ret = []
                for c, m in zip(changed, meta):
                    new_q = re.sub(r'\b%s\b' % re.escape(m[0]), m[1], question.text)
                    ret.append((c, new_q))
                return ret, meta
            return change_both
        t = Perturb.perturb(processed_qs, change_thing(Perturb.change_names), nsamples=1000, meta=True)
        test = INV(**t, name='Change name everywhere', capability='Robustness', description='')
        suite.add(test)

        t = Perturb.perturb(processed_qs, change_thing(Perturb.change_location), nsamples=1000, meta=True)
        test = INV(**t, name='Change location everywhere', capability='Robustness', description='')
        suite.add(test)
        """

        logger.info('constructing test suite complete')
        if only_construct_suite == True:
            return suite
    else:
        suite = given_suite
  
    #from pattern.en import sentiment
    def predict_proba(inputs):
        features = glue_dataset.convert_checklist_input(inputs)
        if verbose == True:
            print('len(inputs)', len(inputs))
            print('debug inputs[0] after conversion', tokenizer.decode(features[0].input_ids))
        model.eval()
        idx_now, bz, probs = 0, 8, []
        while idx_now < len(features):
            input_ids = torch.LongTensor([f.input_ids for f in features[idx_now:idx_now + bz]]).cuda()
            attention_mask = torch.LongTensor([f.attention_mask for f in features[idx_now:idx_now + bz]]).cuda()
            idx_now += bz
            outputs = model(input_ids = input_ids, attention_mask = attention_mask)
            logits = outputs[0]
            prob = torch.softmax(logits, dim = -1).cpu().detach()
            probs.append(prob)

        """
        p1 = np.array([(sentiment(x)[0] + 1)/2. for x in inputs]).reshape(-1, 1)
        p1 = np.random.uniform(size=p1.shape)
        p0 = 1- p1
        pp = np.hstack((p0, p1))
        """
        pp = torch.cat(probs, dim = 0).numpy()
        return pp
    
    from checklist.pred_wrapper import PredictorWrapper
    wrapped_pp = PredictorWrapper.wrap_softmax(predict_proba)
    
    suite.run(wrapped_pp, overwrite=True)
    #suite.summary()
    res = my_summary(suite, verbose = verbose)

    logger.info('do_checklist_SST2 complete')
    return res

def get_funct(task_name):
    cfn = None

    if task_name.lower() == 'qqp':
        cfn = do_checklist_QQP

    if task_name.lower() == 'qnli':
        cfn = do_checklist_QNLI

    if task_name.lower() == 'sst-2':
        cfn = do_checklist_SST2
    
    if cfn == None:
        cfn = None

    return cfn

def construct_checklist_suite(model, tokenizer, eval_dataset, all_args):
    model_args, data_args, training_args, my_args = all_args 
    cfn = get_funct(data_args.task_name)
    
    if cfn is None:
        return None

    suite = cfn(model, tokenizer, eval_dataset, all_args, only_construct_suite = True)
    return suite

def run_checklist_suite(model, tokenizer, eval_dataset, all_args, given_suite = None, verbose = True):
    model_args, data_args, training_args, my_args = all_args 
    cfn = get_funct(data_args.task_name)
    
    if given_suite is not None:
        res = cfn(model, tokenizer, eval_dataset, all_args, only_construct_suite = False, given_suite = given_suite, verbose = verbose)
    else:
        res = {'AVG': 0}
    return res


