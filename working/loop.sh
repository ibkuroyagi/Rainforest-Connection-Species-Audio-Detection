#!/bin/bash

# Copyright 2020 Ibuki Kuroyanagi.
# Created by Ibuki Kuroyanagi
No=v002 #v002-clip065-2
# model=Cnn14_DecisionLevelAtt
type=wave

model=ResNext50
# type=mel128hop1024

stage=2
stop_stage=100
# for No in v000 v001; do
# for checkpoint in best_score checkpoint-1000 checkpoint-2000 checkpoint-3000; do
resume=""
# for fold in {0..4}; do
#     resume+="exp/${type}/${model}/${No}/checkpoint-3000/checkpoint-3000fold${fold}.pkl "
# done
sbatch -J "${type}/${No}" ./run.sh \
    --conf "conf/tuning/${model}.${No}.yaml" \
    --tag "${type}/${model}/${No}" \
    --stage "${stage}" \
    --stop_stage "${stop_stage}" \
    --type "${type}" \
    --cal_type "0" \
    --resume "${resume}" \
    --verbose 1
# --checkpoint "${checkpoint}" \
# done
# done
