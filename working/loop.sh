#!/bin/bash

# Copyright 2020 Ibuki Kuroyanagi.
# Created by Ibuki Kuroyanagi
No=v005 #v003-cos25-center2
model=Cnn14_DecisionLevelAtt
type=wave

# model=ResNext50
# type=mel128hop1024

stage=1
stop_stage=2
# for No in v002-cos25 v004-cos25; do
# for checkpoint in best_score checkpoint-1000 checkpoint-2000; do
sbatch -J "${type}/${No}" ./run.sh \
    --conf "conf/tuning/${model}.${No}.yaml" \
    --tag "${type}/${model}/${No}" \
    --stage "${stage}" \
    --stop_stage "${stop_stage}" \
    --type "${type}" \
    --cache_path "" \
    --cal_type "0" \
    --verbose 1
# done
# done
