#source activate hiddenebm_2006
#source ./add_path.sh
export GLUE_DIR=./data-sets/glue/glue_data/
#export TASK_NAME=QNLI #MRPC #QNLI #RTE #MNLI #SST-2
TASK_NAME=${TASK_NAME:-SST-2}
STEPDIV=${STEPDIV:-4}
STEPMUL=${STEPMUL:-1}
LR=${LR:-2e-5}
#MM=bert-base-uncased
MM=roberta-base 
#MM=albert-base-v2
echo TASK_NAME is $TASK_NAME MM is $MM
sleep 2

#python run_glue.py --model_name_or_path $MM --task_name $TASK_NAME --overwrite_cache --do_train --do_eval --data_dir $GLUE_DIR/$TASK_NAME   --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ../exps/debug_glue_play_$MM/$TASK_NAME/ --overwrite_output_dir

#python run_glue.py --model_name_or_path ../exps/debug_glue_play_$MM/$TASK_NAME --task_name $TASK_NAME --do_eval --data_dir $GLUE_DIR/$TASK_NAME   --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ../exps/debug_glue_play_$MM/$TASK_NAME/

if [ $TASK_NAME == 'SST-2' ]; then
	#LR=1e-5
	BA=32
	MAXSTEP=20935
	WARMSTEP=1256
    #LR=5e-6 #16e-5 #1e-5 #4e-5 #2e-5
    #MAXSTEP=40000 #1250 #2500 #5000 #10500
    #WARMSTEP=1256 #75 #150 #300 #600
fi
if [ $TASK_NAME == 'MNLI' ]; then
	#LR=1e-5
	BA=32
	MAXSTEP=123873
	WARMSTEP=7432
    #LR=5e-6 #1e-5 #4e-5 #2e-5
    #MAXSTEP=240000 #7500 #15000 #31000 #62000
    #WARMSTEP=7432 #450 #900 #7432 #1800 #3700
fi
if [ $TASK_NAME == 'QNLI' ]; then
	#LR=1e-5
	BA=32
	MAXSTEP=33112
	WARMSTEP=1986
    #LR=5e-6 #16e-5 #1e-5 #4e-5 #2e-5
    #MAXSTEP=66000 #2000 #4150 #66000 #8300 #16600
    #WARMSTEP=1986 #125 #250 #1986 #500 #990
fi
if [ $TASK_NAME == 'QQP' ]; then
	#LR=1e-5
	BA=32
	MAXSTEP=113272
	WARMSTEP=28318
	#LR=5e-6 #1e-5 #4e-5 #2e-5 #1e-5
	#BA=32
	#MAXSTEP=220000 #7000 #220000 #56600 #28000 #56600 #113272
	#WARMSTEP=28318 #28318 #14150 #7000 #14150 #28318
fi
if [ $TASK_NAME == 'MRPC' ]; then
	BA=16
	LR=1e-5
	MAXSTEP=2296
	WARMSTEP=137
	STEPDIV=1
fi
if [ $TASK_NAME == 'CoLA' ]; then
	BA=16
	LR=1e-5
	MAXSTEP=5536
	WARMSTEP=320
	STEPDIV=1
fi
if [ $TASK_NAME == 'RTE' ]; then
	BA=16
	LR=2e-5
	MAXSTEP=2036
	WARMSTEP=122
	STEPDIV=1
fi
if [ $TASK_NAME == 'WNLI' ]; then
	BA=16
	LR=1e-5
	MAXSTEP=250
	WARMSTEP=10
	STEPDIV=1
fi

let "MAXSTEP = $MAXSTEP / $STEPDIV "
let "WARMSTEP = $WARMSTEP / $STEPDIV "
let "MAXSTEP = $MAXSTEP * $STEPMUL "
let "WARMSTEP = $WARMSTEP * $STEPMUL "
echo LR: $LR STEPDIV: $STEPDIV MAXSTEP: $MAXSTEP WARMSTEP: $WARMSTEP
OUTDIR=./exps/glue_baseline_$MM/$TASK_NAME/LR${LR}BA${BA}MAXSTEP${MAXSTEP}WARMSTEP${WARMSTEP}
echo OUTDIR is $OUTDIR 
sleep 3

if [ $1 == 'train' ]; then
	echo doing train
	sleep 1
	python nce_glue/run_glue.py --model_name_or_path roberta-base --task_name $TASK_NAME --overwrite_cache --do_train --do_eval_calibration --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_device_train_batch_size $BA --learning_rate $LR --output_dir $OUTDIR --overwrite_output_dir --logging_steps 50 --save_steps -1 --warmup_steps $WARMSTEP --max_steps $MAXSTEP --weight_decay 0.1 --evaluate_during_training --eval_steps 500
	#    python run_glue.py --model_name_or_path $MM --task_name $TASK_NAME --overwrite_cache --do_train --do_eval --do_eval_calibration --do_eval_noise_robustness --data_dir $GLUE_DIR/$TASK_NAME   --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ../exps/glue_play_$MM/$TASK_NAME/  --overwrite_output_dir --logging_steps 50 --save_steps -1
fi

if [ $1 == 'eval' ]; then
    python nce_glue/run_glue.py --model_name_or_path $OUTDIR --task_name $TASK_NAME --do_eval_calibration --data_dir $GLUE_DIR/$TASK_NAME   --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate $LR --output_dir $OUTDIR --overwrite_cache
fi

#python run_glue.py --model_name_or_path ../exps/glue_play_$MM/$TASK_NAME/ --overwrite_cache --task_name $TASK_NAME --do_eval_calibration --data_dir $GLUE_DIR/$TASK_NAME   --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ../exps/glue_play/$TASK_NAME/

#--model_type bert \
