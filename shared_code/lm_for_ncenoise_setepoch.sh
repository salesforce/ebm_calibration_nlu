EP=3
if [ $TASK_NAME == 'MNLI' ]; then
	EP=4
fi
if [ $TASK_NAME == 'QQP' ]; then
	EP=4
fi
LMDIR=./exps/lm_finetune/gpt/${TASK_NAME}/EP${EP}BA24LR5e-06


