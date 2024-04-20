#!/bin/bash

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="svjack/pokemon-blip-captions-en-zh"

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --caption_column="en_text" \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --report_to="wandb" \
  --tracker_project_name="Pokemon-finetung" \
  --validation_prompts="a red smiling pokemon" \
  --validation_epochs=100 \
  --output_dir="sd-poke-2"