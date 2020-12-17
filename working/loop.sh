#!/bin/bash

# Copyright 2020 Ibuki Kuroyanagi.
# Created by Ibuki Kuroyanagi
No=v000
model=Cnn14_DecisionLevelAtt
type=wave
stage=2
stop_stage=2
# for No in v000 v003 v004; do
sbatch -J "${type}/${model}/${No}" ./run.sh \
    --conf "conf/tuning/${model}.${No}.yaml" \
    --tag "${type}/${model}/${No}" \
    --stage "${stage}" \
    --stop_stage "${stop_stage}" \
    --cal_type "0" \
    --verbose 1
# done
