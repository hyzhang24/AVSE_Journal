#!/usr/bin/env bash
cmd="/home3/chenchen/research/maison2/egs/VB/slurm.pl --quiet --nodelist=node09 --gpu 2"

source activate avse_new

$cmd log/train.log \
python train.py \
      --base_dir /home3/hexin/avse_data/ \
      --no_wandb --gpus 2 \
      --inject_type default


