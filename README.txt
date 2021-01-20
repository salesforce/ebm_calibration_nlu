copy /export/share/tianxing-he/lm_finetune_gensave_20200805 to ./exps/lm_finetune
these will have partial2full noise samples with maskrate 0.2 or 0.4 for sst-2, mnli, qqp, qnli
and also labeledH2F nosie samples for mnli, qqp, qnli

i use conda, i use torch==1.5.0, transformers==2.11.0
also you need to install checklist https://github.com/marcotcr/checklist, if you meet problem related to sql during install, check https://github.com/marcotcr/checklist/issues/7

change the paths in ./add_path.sh to your paths

all the scripts are in ./scripts/
my conda env name is hiddenebm_2006, just change it to yours
put the glue data somewhere, and point the GLUE_DIR variable in the scripts to it
if you want to run with roberta-large, change the MM variable in the scripts
note that the nce code is not tested with bert, in that case please do some debug in nce_glue/my_trainer.py and check whether the noise feature input is consistent with the data feature inputs (they should have the same format)

below are example commands:

===LM commands===
MODE=partial2full RATE=0.2 TASK_NAME=QNLI bash ./scripts/lm_ncenoise.sh train 
(this is for training)
MODE=labeledH2F RATE=0.3 TASK_NAME=QNLI bash ./scripts/lm_ncenoise.sh train
(The Rate0.3 in labeled mode is meaningless)
MODE=partial2full RATE=0.2 TASK_NAME=QQP bash ./scripts/lm_ncenoise.sh gen_save
(gen_save is to generate noise samples with a trained model)

===GLUE_NCE commands===
MMODE=partial2full LMP2FRATE=0.2 NMODE=hidden NRATIO=2 TASK_NAME=QNLI ./scripts/glue_nce.sh train
(Parameters to tune: NRATIO(e.g. 1/2/4/8/16/32) and NLAMBDA(e.g. 1/0.5/0.25))
(NMODE=hidden is version 2, NMODE=normal is version 1, NMODE=labeled is version 3)
(You can also do "eval", to test a trained model)

===GLUE_BASELINE commands===
LR=2e-5 STEPDIV=4 TASK_NAME=SST-2 ./scripts/glue_baseline.sh train
(STEPDIV=4 means the MAX_STEP will be recommended step / 4, in general I find that less max_step gives better calibration for the baseline)

Tianxing
