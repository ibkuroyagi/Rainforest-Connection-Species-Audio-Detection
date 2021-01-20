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
# type=raw
# type=mel128hop1024

stage=2
stop_stage=3
verbose=1
No=v032
# for No in v015 v016 v017; do
# for checkpoint in best_score checkpoint-1000 checkpoint-2000 checkpoint-3000 checkpoint-4000; do
# checkpoints="exp/${type}/${model}/${No}/best_score/best_scorefold0.pkl exp/${type}/${model}/${No}/best_score/best_scorefold1.pkl no_model  no_model no_model"
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
    --speed_facters "0.8 0.9 1.1 1.2" \
    --verbose "${verbose}" \
    --cache_path ""
# --checkpoints "${checkpoints}"
# done
# done
