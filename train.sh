

#!/bin/bash

# Get the current date
current_date=$(date)

# Print the current date
echo "The current date is: $current_date"

# Define the arguments for your training script
GPUs="$1"
NUM_GPU=$(echo $GPUs | awk -F, '{print NF}')
NUM_EPOCHS=10
PROJECT_NAME="Frequency-Masking"
MODEL_NAME="RN50" # RN50_mod, RN50, clip_vitl14, clip_rn50
MASK_TYPE="nomask" # nomask, spectral, pixel, patch
BAND="all" # all, low, mid, high
RATIO=15
BATCH_SIZE=128
WANDB_ID="2w0btkas"
RESUME="from_last" # from_last or from_best
learning_rate=0.0002


# Set the CUDA_VISIBLE_DEVICES environment variable to use GPUs
export CUDA_VISIBLE_DEVICES=$GPUs

export TORCH_NCCL_BLOCKING_WAIT=1  # Or 0, depending on your needs

echo "Using $NUM_GPU GPUs with IDs: $GPUs"
#python -m torch.distributed.launch --nproc_per_node=$NUM_GPU train.py \
# Run the distributed training command
torchrun --nproc_per_node=$NUM_GPU train.py \
  --num_epochs $NUM_EPOCHS \
  --project_name $PROJECT_NAME \
  --model_name $MODEL_NAME \
  --mask_type $MASK_TYPE \
  --band $BAND \
  --ratio $RATIO \
  --lr ${learning_rate} \
  --batch_size $BATCH_SIZE \
  --early_stop \
  --pretrained 
  # --clip_grad \
  # --debug \
  # --resume_train $RESUME \
  # --debug \
  # --wandb_online \
  # --wandb_run_id $WANDB_ID \
