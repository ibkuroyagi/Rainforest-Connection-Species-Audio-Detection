#!/bin/bash

# Copyright 2020 Ibuki Kuroyanagi.
# Created by Ibuki Kuroyanagi
No=v002-clip065
model=Cnn14_DecisionLevelAtt
# model=conformer
type=wave
verbose=1
# model=ResNext50
# type=mel128hop1024

stage=2
stop_stage=100
# for model in conformer transformer; do
# for No in v013-sum v013; do
# for checkpoint in best_score checkpoint-1000 checkpoint-2000 checkpoint-3000 checkpoint-4000; do
# resume="exp/${type}/${model}/v012-sum/checkpoint-4000/checkpoint-4000fold0.pkl exp/${type}/${model}/v012-sum/checkpoint-3000/checkpoint-3000fold1.pkl exp/${type}/${model}/v012-sum/checkpoint-1000/checkpoint-1000fold0.pkl exp/${type}/${model}/v012-sum/checkpoint-1000/checkpoint-1000fold0.pkl exp/${type}/${model}/v012-sum/checkpoint-1000/checkpoint-1000fold0.pkl"
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
    --verbose "${verbose}"
# --checkpoint "${checkpoint}"
# done
# done
