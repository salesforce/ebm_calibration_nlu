#source activate hiddenebm_2006
#source ./add_path.sh
export GLUE_DIR=./data-sets/glue/glue_data/
#export TASK_NAME=QNLI #MRPC #QNLI #RTE #MNLI #SST-2
TASK_NAME=${TASK_NAME:-SST-2}
#MM=bert-base-uncased
echo TASK_NAME is $TASK_NAME 
sleep 2

BA=24
LEN=128
LR=${LR:-5e-06}
SEED=${SEED:-42}
EP=${EP:-4}
LOGSTEP=${LOGSTEP:-20000}
#MODE=partial2full
MODE=${MODE:-labeledP2F} #labeledP2F or partial2full or labeledH2F
RATE=${RATE:-0.3}

P2FLABELTEMP=1.0
#if [[ $TASK_NAME == 'SST-2' ]]; then
#    P2FLABELTEMP=2.0
#fi

if [[ $TASK_NAME == 'MNLI' || $TASK_NAME == 'QQP' || $TASK_NAME == 'QNLI' ]]; then
    P2FLABELTEMP=1.0
    echo setting P2FLABELTEMP to $P2FLABELTEMP
fi

DRYRUN=${DRYRUN:-False}
B_CORR=${B_CORR:-False}
B_PATH=None
if [[ $TASK_NAME == 'SST-2' ]]; then
    B_PATH=./exps/glue_baseline_roberta-base/SST-2/LR2e-5BA32MAXSTEP5000WARMSTEP300
fi
if [[ $TASK_NAME == 'MNLI' ]]; then
    B_PATH=./exps/glue_baseline_roberta-base/MNLI/LR2e-5BA32MAXSTEP31000WARMSTEP1800
fi
if [[ $TASK_NAME == 'QNLI' ]]; then
    B_PATH=./exps/glue_baseline_roberta-base/QNLI/LR2e-5BA32MAXSTEP8300WARMSTEP500
fi
if [[ $TASK_NAME == 'QQP' ]]; then
    B_PATH=./exps/glue_baseline_roberta-base/QQP/LR2e-5BA32MAXSTEP28318WARMSTEP7079
fi

if [[ $MODE == 'partial2full' || $MODE == 'labeledP2F' || $MODE == 'labeledH2F' || $MODE == 'H2Flabeled' ]]; then
    LEN=256
    BA=8
    echo setting LEN to $LEN BA to $BA
fi
if [[ $MODE == 'H2Flabeled' ]]; then
    echo H2Flabeled, force setting RATE to 0
    RATE=0
    sleep 1
fi

if [[ $TASK_NAME == 'RTE' || $TASK_NAME == 'CoLA' ]]; then
    echo RTE or CoLA, setting EP to 8
    EP=8
fi
if [[ $TASK_NAME == 'WNLI' ]]; then
    echo WNLI, setting EP to 16
    EP=16
fi



OUTDIR=./exps/lm_finetune/gpt_MODE${MODE}/${TASK_NAME}/EP${EP}BA${BA}LR${LR}P2FRATE${RATE}

echo MODE is $MODE
echo $OUTDIR

sleep 2

if [ $1 == 'train' ]; then
	echo do_train
	sleep 1
	python lm_for_gluenoise/run_language_modeling.py --output_dir=$OUTDIR --model_type=gpt2 --model_name_or_path=gpt2 --do_train --do_eval --task_name $TASK_NAME --overwrite_cache --max_seq_length $LEN --overwrite_output_dir --evaluate_during_training --data_dir $GLUE_DIR/$TASK_NAME --per_device_train_batch_size $BA --num_train_epochs $EP --learning_rate $LR --logging_steps ${LOGSTEP} --save_steps -1 --noiselm_mode $MODE --noiselm_partial2full_maskrate $RATE --noiselm_partial2full_prefixlen 128
fi

if [ $1 == 'eval' ]; then
	echo do_eval
	sleep 1
	python lm_for_gluenoise/run_language_modeling.py --output_dir=$OUTDIR --model_type=gpt2 --model_name_or_path=$OUTDIR --do_eval --task_name $TASK_NAME --overwrite_cache --max_seq_length $LEN --overwrite_output_dir --evaluate_during_training --data_dir $GLUE_DIR/$TASK_NAME --per_device_train_batch_size $BA --num_train_epochs $EP --learning_rate $LR --logging_steps ${LOGSTEP} --save_steps -1 --noiselm_mode $MODE --noiselm_partial2full_maskrate $RATE --noiselm_partial2full_prefixlen 128
fi
#
#only for WNLI
#python run_language_modeling.py --output_dir=../exps/lm_finetune/gpt/${TASK_NAME}/BA${BA}LR${LR} --model_type=gpt2 --model_name_or_path=gpt2 --do_train --do_eval --task_name $TASK_NAME --overwrite_cache --max_seq_length $LEN --overwrite_output_dir --evaluate_during_training --data_dir $GLUE_DIR/$TASK_NAME --per_device_train_batch_size $BA --num_train_epochs 30.0 --learning_rate $LR --logging_steps 10 --save_steps 10000

#EP=3
#if [ $TASK_NAME == 'MNLI' ]; then
#	EP=4
#fi
#if [ $TASK_NAME == 'QQP' ]; then
#	EP=4
#fi
#source shared_code/lm_for_ncenoise_setepoch.sh #now all epoch are from EP4
#OUTDIR=$LMDIR
sleep 1

if [ $1 == 'gen_corr' ]; then
	echo do_gen_save seed forced to be 42 and 1
	echo $OUTDIR
	sleep 2
	python lm_for_gluenoise/run_language_modeling.py --output_dir=$OUTDIR --model_name_or_path=$OUTDIR --do_eval --do_gen_save --task_name $TASK_NAME --max_seq_length $LEN --data_dir $GLUE_DIR/$TASK_NAME --gen_save_number 50000 --seed 43 --noiselm_mode $MODE --noiselm_partial2full_maskrate $RATE --noiselm_partial2full_prefixlen 128 --noiselm_labeledP2F_labelsample_temperature $P2FLABELTEMP --overwrite_cache --dry_run --bert_baseline_path $B_PATH --do_baseline_correlate
fi

if [ $1 == 'gen_save' ]; then
	echo do_gen_save seed forced to be 42 and 1
	echo $OUTDIR
	sleep 2
	python lm_for_gluenoise/run_language_modeling.py --output_dir=$OUTDIR --model_name_or_path=$OUTDIR --do_eval --do_gen_save --task_name $TASK_NAME --max_seq_length $LEN --data_dir $GLUE_DIR/$TASK_NAME --gen_save_number 50000 --seed 42 --noiselm_mode $MODE --noiselm_partial2full_maskrate $RATE --noiselm_partial2full_prefixlen 128 --noiselm_labeledP2F_labelsample_temperature $P2FLABELTEMP --overwrite_cache 
	python lm_for_gluenoise/run_language_modeling.py --output_dir=$OUTDIR --model_name_or_path=$OUTDIR --do_eval --do_gen_save --task_name $TASK_NAME --max_seq_length $LEN --data_dir $GLUE_DIR/$TASK_NAME --gen_save_number 2000000 --seed 1 --noiselm_mode $MODE --noiselm_partial2full_maskrate $RATE --noiselm_partial2full_prefixlen 128 --noiselm_labeledP2F_labelsample_temperature $P2FLABELTEMP --overwrite_cache 
	#python lm_for_gluenoise/run_language_modeling.py --output_dir=$OUTDIR --model_name_or_path=$OUTDIR --do_eval --do_gen_save --task_name $TASK_NAME --max_seq_length $LEN --data_dir $GLUE_DIR/$TASK_NAME --gen_save_number 2000000 --seed 2 --noiselm_mode $MODE --noiselm_partial2full_maskrate $RATE --noiselm_partial2full_prefixlen 128 --noiselm_labeledP2F_labelsample_temperature $P2FLABELTEMP --overwrite_cache
fi
 
if [ $1 == 'gen_save_seed' ]; then
	echo do_gen_save_seed seed is $SEED
	echo $OUTDIR
	sleep 2
	python lm_for_gluenoise/run_language_modeling.py --output_dir=$OUTDIR --model_name_or_path=$OUTDIR --do_eval --do_gen_save --task_name $TASK_NAME --max_seq_length $LEN --data_dir $GLUE_DIR/$TASK_NAME --gen_save_number 2000000 --seed $SEED --noiselm_mode $MODE --noiselm_partial2full_maskrate $RATE --noiselm_partial2full_prefixlen 128 --noiselm_labeledP2F_labelsample_temperature $P2FLABELTEMP --overwrite_cache
fi
 
    #--output_dir../exps/noise_lm_finetune_gpt2/$TASK_NAME/

#python run_glue.py --model_name_or_path $MM --task_name $TASK_NAME --overwrite_cache --do_train --do_eval --data_dir $GLUE_DIR/$TASK_NAME   --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ../exps/glue_play_$MM/$TASK_NAME/
#python run_glue.py --model_name_or_path ../exps/glue_play/$TASK_NAME --task_name $TASK_NAME --do_eval --data_dir $GLUE_DIR/$TASK_NAME   --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ../exps/glue_play/$TASK_NAME/
#python run_glue.py --model_name_or_path ../exps/glue_play_$MM/$TASK_NAME/ --overwrite_cache --task_name $TASK_NAME --do_eval_calibration --data_dir $GLUE_DIR/$TASK_NAME   --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ../exps/glue_play/$TASK_NAME/

#--model_type bert \
