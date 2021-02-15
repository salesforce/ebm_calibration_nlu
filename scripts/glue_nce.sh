#source activate hiddenebm_2006
#source ./add_path.sh
export GLUE_DIR=./data-sets/glue/glue_data/
#export TASK_NAME=QNLI #MRPC #QNLI #RTE #MNLI #SST-2
TASK_NAME=${TASK_NAME:-SST-2} LEN=128
#MM=bert-base-uncased
MM=roberta-base 
EP=${EP:-3.0}
#MM=albert-base-v2
LR=${LR:-2e-5}
BZ=${BZ:-32}
NBZ=${NBZ:-32}
NLAMBDA=${NLAMBDA:-1}
NRATIO=${NRATIO:-1}
NMODE=${NMODE:-normal} #normal or hidden or labeled

SEED=${SEED:-42}
LMMODE=${LMMODE:-labeledP2F} #partial2full
LMP2FRATE=${LMP2FRATE:-0.3}
LMSAMPLE=TOPK20
STEPDIV=${STEPDIV:-4}
DEBUG=${DEBUG:-0}

echo TASK_NAME is $TASK_NAME MM is $MM NMODE is $NMODE 
echo DEBUG is $DEBUG
sleep 2

#source shared_code/lm_for_ncenoise_setepoch.sh
#echo LMDIR is $LMDIR
#NF=$LMDIR/gen_saves/${NSAMPLE}_num250000_seed1.txt 
#NF2=$LMDIR/gen_saves/${NSAMPLE}_num250000_seed42.txt
P2FLABELTEMP=1.0
#if [[ $TASK_NAME == 'SST-2' ]]; then
#    P2FLABELTEMP=2.0
#    echo SST-2, setting P2FLABELTEMP to $P2FLABELTEMP
#fi
LMEP=4
if [[ $TASK_NAME == 'RTE' || $TASK_NAME == 'CoLA' || $TASK_NAME == 'MRPC' ]]; then
    echo RTE or CoLA or MRPC, setting LMEP to 8
    LMEP=8
fi
if [[ $TASK_NAME == 'WNLI' ]]; then
    echo WNLI, setting LMEP to 16
    LMEP=16
fi

if [[ $LMMODE == 'partial2full' ]]; then
    NF=./exps/lm_finetune/gpt_MODE${LMMODE}/${TASK_NAME}/EP${LMEP}BA8LR5e-06P2FRATE${LMP2FRATE}/gen_saves/${LMSAMPLE}_num2000000_seed1.txt #${LMSAMPLE}_num500000_seed1.txt 
    NF2=./exps/lm_finetune/gpt_MODE${LMMODE}/${TASK_NAME}/EP${LMEP}BA8LR5e-06P2FRATE${LMP2FRATE}/gen_saves/${LMSAMPLE}_num50000_seed42.txt
    #if [[ $TASK_NAME == 'MRPC' ]]; then
    #    NF=./exps/glue_mlm_roberta-base/MRPC/notrain_rate0/gen_saves/TOPK10_num2000000_seed1.txt
    #	NF2=./exps/glue_mlm_roberta-base/MRPC/notrain_rate0/gen_saves/TOPK10_num50000_seed42.txt
    #fi
fi
if [[ $LMMODE == 'mlm' ]]; then
    NF=./exps/glue_mlm_roberta-base/${TASK_NAME}/notrain_rate${LMP2FRATE}/gen_saves/TOPK10_num2000000_seed1.txt
    NF2=./exps/glue_mlm_roberta-base/${TASK_NAME}/notrain_rate${LMP2FRATE}/gen_saves/TOPK10_num50000_seed42.txt
fi
if [[ $LMMODE == 'labeledP2F' || $LMMODE == 'labeledH2F' ]]; then
    NF=./exps/lm_finetune/gpt_MODE${LMMODE}/${TASK_NAME}/EP${LMEP}BA8LR5e-06P2FRATE${LMP2FRATE}/gen_saves/${LMSAMPLE}_num2000000_seed1_labeltemp${P2FLABELTEMP}.txt
    NF2=./exps/lm_finetune/gpt_MODE${LMMODE}/${TASK_NAME}/EP${LMEP}BA8LR5e-06P2FRATE${LMP2FRATE}/gen_saves/${LMSAMPLE}_num50000_seed42_labeltemp${P2FLABELTEMP}.txt
fi

echo NF is $NF
echo NF2 is $NF2

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
	BZ=16
	NBZ=16
	LR=1e-5
	MAXSTEP=2296
	WARMSTEP=137
	STEPDIV=1
fi
if [ $TASK_NAME == 'CoLA' ]; then
	BA=16
	BZ=16
	NBZ=16
	LR=1e-5
	MAXSTEP=5536
	WARMSTEP=320
	STEPDIV=1
fi
if [ $TASK_NAME == 'RTE' ]; then
	BA=16
	BZ=16
	NBZ=16
	LR=2e-5
	MAXSTEP=2036
	WARMSTEP=122
	STEPDIV=1
fi
if [ $TASK_NAME == 'WNLI' ]; then
	BA=16
	BZ=16
	NBZ=16
	LR=1e-5
	MAXSTEP=250
	WARMSTEP=10
	STEPDIV=1
fi


let "MAXSTEP = $MAXSTEP / $STEPDIV "
let "WARMSTEP = $WARMSTEP / $STEPDIV "
echo LR: $LR STEPDIV: $STEPDIV MAXSTEP: $MAXSTEP WARMSTEP: $WARMSTEP


#OUTDIR=../exps/glue_baseline_$MM/$TASK_NAME/LR${LR}BA${BA}MAXSTEP${MAXSTEP}WARMSTEP${WARMSTEP}
OUTDIR=./exps/glue_nceM${NMODE}_noiselmM${LMMODE}_${MM}/$TASK_NAME/LR${LR}BA${BZ}MAXSTEP${MAXSTEP}WARMSTEP${WARMSTEP}LMNSAMPLE${LMSAMPLE}MODE${LMMODE}P2FRATE${LMP2FRATE}NCEMODE${NMODE}LAMBDA${NLAMBDA}RATIO${NRATIO} 
echo NF is $NF LMSAMPLE is $LMSAMPLE LMMODE is $LMMODE
echo NMODE is $NMODE
sleep 1
echo OUTDIR is $OUTDIR
sleep 1

if [[ $1 == 'debug' ]]; then
	OUTDIR=./exps/glue_nce_${MM}/$TASK_NAME/debug
	echo doing debug, OUTDIR is $OUTDIR
	sleep 1	
	python nce_glue/run_glue.py --model_name_or_path $MM --task_name $TASK_NAME --do_train --do_eval --do_eval_calibration --do_eval_noise_robustness --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length $LEN --per_device_train_batch_size $BZ --learning_rate $LR --output_dir $OUTDIR --overwrite_output_dir --nce_noise_file $NF --nce_noise_eval_file $NF2 --cache_dir ./cache_data/nce_debug/ --train_mode nce_noise --nce_noise_ratio ${NRATIO} --nce_noise_batch_size $NBZ --nce_mode $NMODE --logging_steps 50 --save_steps -1 --eval_steps 500 --nce_lambda $NLAMBDA --evaluate_during_training --warmup_steps $WARMSTEP --max_steps $MAXSTEP --weight_decay 0.1 --noiselm_mode $LMMODE
	#python nce_glue/run_glue.py --model_name_or_path $OUTDIR --task_name $TASK_NAME --do_eval --do_eval_calibration --do_eval_noise_robustness --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length $LEN --per_device_train_batch_size $BZ --learning_rate $LR --output_dir $OUTDIR --overwrite_output_dir --nce_noise_file $NF --nce_noise_eval_file $NF2 --cache_dir ./cache_data/nce_debug/ --train_mode nce_noise --nce_noise_ratio ${NRATIO} --nce_noise_batch_size $NBZ --nce_mode $NMODE --logging_steps 50 --save_steps -1 --eval_steps 500 --nce_lambda $NLAMBDA --evaluate_during_training --warmup_steps $WARMSTEP --max_steps $MAXSTEP --weight_decay 0.1  --overwrite_cache
fi

if [ $1 == 'train_debug' ]; then
	echo doing train_debug
	OUTDIR=./exps/debug/glue_nceM${NMODE}_noiselmM${LMMODE}_${MM}/$TASK_NAME/LR${LR}BA${BZ}MAXSTEP${MAXSTEP}WARMSTEP${WARMSTEP}LMNSAMPLE${LMSAMPLE}MODE${LMMODE}P2FRATE${LMP2FRATE}NCEMODE${NMODE}LAMBDA${NLAMBDA}RATIO${NRATIO} 
	echo OUTDIR is $OUTDIR	
	sleep 1
	python nce_glue/run_glue.py --model_name_or_path $MM --task_name $TASK_NAME --do_train --do_eval --do_eval_calibration --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length $LEN --per_device_train_batch_size $BZ --learning_rate $LR --output_dir $OUTDIR --overwrite_output_dir --nce_noise_file $NF --nce_noise_eval_file $NF2 --cache_dir ./cache_data/nce/ --train_mode nce_noise --nce_noise_ratio ${NRATIO} --nce_noise_batch_size $NBZ --nce_mode $NMODE --logging_steps 50 --save_steps -1 --eval_steps 500 --nce_lambda $NLAMBDA --evaluate_during_training --warmup_steps $WARMSTEP --max_steps $MAXSTEP --weight_decay 0.1 --overwrite_cache --fast_debug $DEBUG --noiselm_mode $LMMODE
fi

if [ $1 == 'train' ]; then
	echo doing train
	sleep 1
	python nce_glue/run_glue.py --model_name_or_path $MM --task_name $TASK_NAME --do_train --do_eval --do_eval_calibration --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length $LEN --per_device_train_batch_size $BZ --learning_rate $LR --output_dir $OUTDIR --overwrite_output_dir --nce_noise_file $NF --nce_noise_eval_file $NF2 --cache_dir ./cache_data/nce/ --train_mode nce_noise --nce_noise_ratio ${NRATIO} --nce_noise_batch_size $NBZ --nce_mode $NMODE --logging_steps 50 --save_steps -1 --eval_steps 500 --nce_lambda $NLAMBDA --evaluate_during_training --warmup_steps $WARMSTEP --max_steps $MAXSTEP --weight_decay 0.1 --overwrite_cache --fast_debug $DEBUG --noiselm_mode $LMMODE
fi

if [ $1 == 'eval' ]; then
	echo doing eval SEED: $SEED
	sleep 1
	python nce_glue/run_glue.py --model_name_or_path $OUTDIR --task_name $TASK_NAME --do_eval_calibration --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length $LEN --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir $OUTDIR --nce_noise_file $NF --nce_noise_eval_file $NF2 --cache_dir ./cache_data/nce/ --train_mode nce_noise --nce_noise_ratio ${NRATIO} --nce_noise_batch_size $NBZ --nce_mode $NMODE --overwrite_cache --seed $SEED --noiselm_mode $LMMODE
fi

if [ $1 == 'energy' ]; then
	echo doing eval SEED: $SEED
	sleep 1
	python nce_glue/run_glue.py --model_name_or_path $OUTDIR --task_name $TASK_NAME --do_eval_calibration --do_energy_analysis --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length $LEN --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir $OUTDIR --nce_noise_file $NF --nce_noise_eval_file $NF2 --cache_dir ./cache_data/nce/ --train_mode nce_noise --nce_noise_ratio ${NRATIO} --nce_noise_batch_size $NBZ --nce_mode $NMODE --overwrite_cache --seed $SEED --noiselm_mode $LMMODE
fi


#python run_glue.py --model_name_or_path ../exps/glue_play_$MM/$TASK_NAME/ --overwrite_cache --task_name $TASK_NAME --do_eval_calibration --data_dir $GLUE_DIR/$TASK_NAME   --max_seq_length $LEN --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ../exps/glue_play/$TASK_NAME/

#--model_type bert \
