


# Joint Energy-based Model Training for Better Calibrated Natural Language Understanding Models

## Introduction

This repo contains code for "Joint Energy-based Model Training for Better Calibrated Natural Language Understanding Models".

Our paper will appear in EACL 2021.

Paper link: https://arxiv.org/pdf/2101.06829.pdf

## Installation

We use the following packages, we recommend you to use a virtual environment (e.g. conda).
- Python 3.7
- torch == 1.5.0
- transformers == 2.11.0
  
Also, please run:
```pip install -r requirements.txt```

## Usage
This section explains steps to preprocess MultiWOZ dataset and training the model. 

### Preprocessing: 
It includes downloading MultiWOZ dataset, performing delexicaliztion, and creating dataset for language model
```
create_dataset.sh
```
Each dialogue turn will be represented as a sequence, which contains previous user/system turns, belief, action, and delexicalized response

```
<|endoftext|> <|context|> <|user|> i am looking for a college type attraction . <|system|> there are 18 colleges i have found , would you prefer 1 in town centre or in the west ? <|user|> i would like to visit on in town centre please . <|system|> sure , we have thirteen options , 10 of which are free . may i suggest king s college , or hughes hall ? <|user|> okay , may i have their postcode , entrance fee , and phone number ?<|endofcontext|> 
<|belief|> attraction type college , attraction name kings college|hughes hall , attraction area centre <|endofbelief|> 
<|action|> attraction inform name , attraction inform fee , attraction inform post , attraction inform phone <|endofaction|> 
<|response|> sure , the post code to [attraction_name] is [attraction_postcode] , the entrance fee is free , and phone number [attraction_phone] <|endofresponse|> <|endoftext|>
```


## Citation

```
@misc{he2021joint,
      title={Joint Energy-based Model Training for Better Calibrated Natural Language Understanding Models}, 
      author={Tianxing He and Bryan McCann and Caiming Xiong and Ehsan Hosseini-Asl},
      year={2021},
      eprint={2101.06829},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


## License

Please see LICENSE.md .

copy /export/share/tianxing-he/lm_finetune_gensave_20200805 to ./exps/lm_finetune
these will have partial2full noise samples with maskrate 0.2 or 0.4 for sst-2, mnli, qqp, qnli
and also labeledH2F nosie samples for mnli, qqp, qnli

i use conda, i use torch==1.5.0, transformers==2.11.0
also, please run
pip install -r transformers_examples_requirements.txt

for glue data, go to ebm_calibration_nlu/data-sets/glue, and run
python download_glue_data.py

all the scripts are in ./scripts/

below are example commands:

===GLUE baseline===
TASK_NAME=SST-2 ./scripts/glue_baseline.sh train

===LM commands===
MODE=partial2full RATE=0.4 TASK_NAME=SST-2 bash ./scripts/lm_ncenoise.sh train 
(this is for training lm)
MODE=partial2full RATE=0.4 TASK_NAME=SST-2 bash ./scripts/lm_ncenoise.sh gen_save
(gen_save is to generate noise samples with a trained model)

===GLUE_NCE commands===
MMODE=partial2full LMP2FRATE=0.2 NMODE=hidden NRATIO=2 TASK_NAME=QNLI ./scripts/glue_nce.sh train
(Parameters to tune: NRATIO(e.g. 1/2/4/8/16/32) and NLAMBDA(e.g. 1/0.5/0.25))
(NMODE=normal is the scalar version, NMODE=hidden is the hidden version, NMODE=selflabeled is the s-hidden version)
(You can also do "eval", to test a trained model)

===GLUE_BASELINE commands===
LR=2e-5 STEPDIV=4 TASK_NAME=SST-2 ./scripts/glue_baseline.sh train
(STEPDIV=4 means the MAX_STEP will be recommended step / 4, in general I find that less max_step gives better calibration for the baseline)


Tianxing
