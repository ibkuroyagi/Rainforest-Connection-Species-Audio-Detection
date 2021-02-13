#!/bin/bash

# Copyright 2020 Ibuki Kuroyanagi.
# Created by Ibuki Kuroyanagi

# model=Cnn14_DecisionLevelAtt
model=EfficientNet
# model=MobileNetV2
# model=ResNext50
# model=conformer
# model=transformer

type=wave
# type=mel256wave
n_jobs=16
n_gpus=1
stage=2
stop_stage=3
verbose=1
No=v043
step=6000
# for No in v027 v028; do
# for checkpoint in best_score checkpoint-1000 checkpoint-2000 checkpoint-3000 checkpoint-4000; do
# resume="exp/${type}/${model}/${No}/best_score/best_scorefold0.pkl no_model no_model  no_model no_model"
# resume="exp/${type}/${model}/${No}/checkpoint-${step}/checkpoint-${step}fold0.pkl no_model no_model no_model no_model"
resume=""
# for fold in {0..4}; do
#     # resume+="exp/${type}/${model}/${No}/checkpoint-${step}/checkpoint-${step}fold${fold}.pkl "
#     resume+="exp/${type}/${model}/${No}/best_score/best_scorefold${fold}.pkl "
# done
sbatch -J "${type}/${No}" ./run.sh \
    --conf "conf/tuning/${model}.${No}.yaml" \
    --n_jobs "${n_jobs}" \
    --n_gpus "${n_gpus}" \
    --tag "${type}/${model}/${No}" \
    --stage "${stage}" \
    --stop_stage "${stop_stage}" \
    --type "${type}" \
    --cal_type 0 \
    --resume "${resume}" \
    --speed_facters "0.9 1.1 0.8 1.2" \
    --verbose "${verbose}" \
    --cache_path ""
# --checkpoints "${checkpoints}"
# done
# done
# 0.8 0.9 1.1 1.2
