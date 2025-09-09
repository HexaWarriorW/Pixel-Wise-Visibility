#! /bin/bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 train_ddp.py \
    --save_path checkpoint_adamw \
    --root_path /media/arthurthomas/Data/dataset/fog_dataset/train/ \
    --batch_size 4 \
    --num_epochs 80