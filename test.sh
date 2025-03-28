#!/bin/bash
# Define the arguments for your test script
GPUs="$1"
NUM_GPU=$(echo $GPUs | awk -F, '{print NF}')
DATA_TYPE="Wang_CVPR20"  # Wang_CVPR20 or Ojha_CVPR23
MODEL_NAME="RN50" # # RN50_mod, RN50, clip_vitl14, clip_rn50
MASK_TYPE="nomask" # spectral, pixel, patch or nomask
BAND="all" # all, low, mid, high
RATIO=15
BATCH_SIZE=64
#export CUDA_VISIBLE_DEVICES=$GPUs # Set the CUDA_VISIBLE_DEVICES environment variable to use GPUs
#export TORCH_NCCL_BLOCKING_WAIT=1  # Or 0, depending on your needs
echo "Using $NUM_GPU GPUs with IDs: $GPUs"
# Run the test command
#python -m torch.distributed.launch --nproc_per_node=$NUM_GPU test.py \
python  test.py \
  --data_type $DATA_TYPE \
  --pretrained \
  --model_name $MODEL_NAME \
  --mask_type $MASK_TYPE \
  --band $BAND \
  --ratio $RATIO \
  --batch_size $BATCH_SIZE \
  --clip_ft \
  # --other_model

