source activate hiddenebm_2006
source ./add_path.sh
export GLUE_DIR=~/data-sets/glue/glue_data/
#export TASK_NAME=QNLI #MRPC #QNLI #RTE #MNLI #SST-2
TASK_NAME=${TASK_NAME:-SST-2}
STEPDIV=${STEPDIV:-4}
STEPMUL=${STEPMUL:-1}
LR=${LR:-2e-5}
#MM=bert-base-uncased
MM=roberta-base 
LMP2FRATE=${LMP2FRATE:-0.15}
#MM=albert-base-v2
echo TASK_NAME is $TASK_NAME MM is $MM
sleep 2

#python run_glue.py --model_name_or_path $MM --task_name $TASK_NAME --overwrite_cache --do_train --do_eval --data_dir $GLUE_DIR/$TASK_NAME   --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ../exps/debug_glue_play_$MM/$TASK_NAME/ --overwrite_output_dir

#python run_glue.py --model_name_or_path ../exps/debug_glue_play_$MM/$TASK_NAME --task_name $TASK_NAME --do_eval --data_dir $GLUE_DIR/$TASK_NAME   --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ../exps/debug_glue_play_$MM/$TASK_NAME/

OUTDIR=./exps/glue_mlm_$MM/$TASK_NAME/notrain_rate${LMP2FRATE}
echo OUTDIR is $OUTDIR 
echo LMP2FRATE is $LMP2FRATE
sleep 2

if [[ $1 == 'gen_save' ]]; then
	echo doing gen_save
	python nce_glue/run_mlm.py --model_name_or_path roberta-base --task_name $TASK_NAME --overwrite_cache --do_train --do_eval_calibration --do_eval_scaling_binning_calibration --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate $LR --output_dir $OUTDIR --overwrite_output_dir --logging_steps 50 --save_steps -1 --weight_decay 0.1 --evaluate_during_training --eval_steps 500 --seed 42 --do_gen_save --gen_save_number 50000 --noiselm_partial2full_maskrate $LMP2FRATE
	python nce_glue/run_mlm.py --model_name_or_path roberta-base --task_name $TASK_NAME --overwrite_cache --do_train --do_eval_calibration --do_eval_scaling_binning_calibration --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_device_train_batch_size 32 --learning_rate $LR --output_dir $OUTDIR --overwrite_output_dir --logging_steps 50 --save_steps -1 --weight_decay 0.1 --evaluate_during_training --eval_steps 500 --seed 1 --do_gen_save --gen_save_number 2000000 --noiselm_partial2full_maskrate $LMP2FRATE
fi


