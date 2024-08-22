#!/bin/bash
set -e

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>

SEED=${1:--1}

MAX_EPOCHS=4000
QUALITY_THRESHOLD="0.90"      #0.908 #0.86170 #8824
START_EVAL_AT=500
START_EVAL_AT=20
EVALUATE_EVERY=20
LEARNING_RATE="0.8"
LR_WARMUP_EPOCHS=200
# SAVE_CKPT_PATH="/DATA/soosung/unet3d/checkpoint_880.pth"  # with shift
SAVE_CKPT_PATH="/DATA/soosung/unet3d/checkpoint_900.pth"  # without shift
DATASET_DIR="/DATA/soosung/unet3d/dataset/kits19"
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=1
EXEC_MODE="train"



if [ -d ${DATASET_DIR} ]
then
    # start timing
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"

# CLEAR YOUR CACHE HERE
  python -c "
from mlperf_logging.mllog import constants
from runtime.logging import mllog_event
mllog_event(key=constants.CACHE_CLEAR, value=True)"

python main.py --data_dir ${DATASET_DIR} \
    --epochs ${MAX_EPOCHS} \
    --evaluate_every ${EVALUATE_EVERY} \
    --start_eval_at ${START_EVAL_AT} \
    --quality_threshold ${QUALITY_THRESHOLD} \
    --batch_size ${BATCH_SIZE} \
    --optimizer sgd \
    --ga_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --seed ${SEED} \
    --lr_warmup_epochs ${LR_WARMUP_EPOCHS} \
    --exec_mode ${EXEC_MODE}\
    --save_ckpt_path ${SAVE_CKPT_PATH} \
    # --pretrained_weights ${PRETRAINED_CKPT_PATH}


	# end timing
	end=$(date +%s)
	end_fmt=$(date +%Y-%m-%d\ %r)
	echo "ENDING TIMING RUN AT $end_fmt"


	# report result
	result=$(( $end - $start ))
	result_name="image_segmentation"


	echo "RESULT,$result_name,$SEED,$result,$USER,$start_fmt"
else
	echo "Directory ${DATASET_DIR} does not exist"
fi