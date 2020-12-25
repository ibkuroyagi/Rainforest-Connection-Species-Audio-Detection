#!/bin/bash

# Copyright 2020 Ibuki Kuroyanagi.
# Created by Ibuki Kuroyanagi
No=v003-aug #v003-cos25-center2
model=Cnn14_DecisionLevelAtt
# type=wave

# model=ResNext50
type=mel128hop1024

stage=1
stop_stage=100
# for No in v000 v001; do
# for checkpoint in best_score checkpoint-1000 checkpoint-2000 checkpoint-3000; do
sbatch -J "${type}/${No}" ./run.sh \
    --conf "conf/tuning/${model}.${No}.yaml" \
    --tag "${type}/${model}/${No}" \
    --stage "${stage}" \
    --stop_stage "${stop_stage}" \
    --type "${type}" \
    --cache_path "" \
    --cal_type "0" \
    --verbose 1
# --checkpoint "${checkpoint}" \
# done
# done
