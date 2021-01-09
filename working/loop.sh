#!/bin/bash

# Copyright 2020 Ibuki Kuroyanagi.
# Created by Ibuki Kuroyanagi

# model=Cnn14_DecisionLevelAtt
# model=conformer
model=EfficientNet
type=wave
verbose=1
# type=mel128hop1024

stage=2
stop_stage=100
No=v011
# for model in conformer transformer; do
# for No in v006 v007; do
# for checkpoint in best_score checkpoint-1000 checkpoint-2000 checkpoint-3000 checkpoint-4000; do
# checkpoints="exp/${type}/${model}/${No}/best_score/best_scorefold0.pkl exp/${type}/${model}/${No}/best_score/best_scorefold1.pkl no_model  no_model no_model"
resume=""
# for fold in {0..4}; do
#     resume+="exp/${type}/${model}/${No}/checkpoint-2000/checkpoint-2000fold${fold}.pkl "
# done
sbatch -J "${type}/${No}" ./run.sh \
    --conf "conf/tuning/${model}.${No}.yaml" \
    --tag "${type}/${model}/${No}" \
    --stage "${stage}" \
    --stop_stage "${stop_stage}" \
    --type "${type}" \
    --cal_type "0" \
    --resume "${resume}" \
    --speed_facters "0.9 1.1" \
    --verbose "${verbose}"
# --checkpoints "${checkpoints}"
# done
# done
