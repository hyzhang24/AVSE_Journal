#!/usr/bin/env bash

cmd="/home3/chenchen/research/maison2/egs/VB/slurm.pl --quiet --nodelist=node03 --gpu 1"
# source activate espnet

$cmd log/train.log \
python try.py