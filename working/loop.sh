#!/bin/bash

# Copyright 2020 Ibuki Kuroyanagi.
# Created by Ibuki Kuroyanagi
# No=v002-clip065
# model=Cnn14_DecisionLevelAtt
# model=conformer
model=EfficientNet
type=wave
verbose=1
# type=mel128hop1024

stage=2
stop_stage=100
No=v005
# for model in conformer transformer; do
for No in v006 v007; do
    # for checkpoint in best_score checkpoint-1000 checkpoint-2000 checkpoint-3000 checkpoint-4000; do
    # resume="exp/${type}/${model}/${No}/best_score/best_scorefold0.pkl no_model no_model  no_model no_model"
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
        --speed_facters "" \
        --verbose "${verbose}"
    # --checkpoint "${checkpoint}"
    # done
done

du -s -h
for step in 1000 2000 3000 4000 5000; do
    for tag in v000 v002 v002-clip035 v002-clip065-cos v002-clip065-original v002-cos25 v002-cos25-center2 v005-red v007 v009-red v011 v012-sum v013-sum v001 v002-binary v002-clip065 v002-clip065-noaug v002-clip065-original2 v002-cos25-center v005 v006 v009 v010 v012 v013; do
        rm -rf "exp/wave/Cnn14_DecisionLevelAtt/${tag}/checkpoint-${step}"
    done
    for tag in v000 v001 v002 v003; do
        rm -rf "exp/wave/ResNext50/${tag}/checkpoint-${step}"
    done
    for tag in v000 v000-sum v002 v001; do
        rm -rf "exp/wave/conformer/${tag}/checkpoint-${step}"
    done
    for tag in v000 v000-sum v002 v001; do
        rm -rf "exp/wave/transformer/${tag}/checkpoint-${step}"
    done
    for tag in v003 v003-aug v003-cos v003-cos25 v003-cos25-center v003-cos25-center2 v008; do
        rm -rf "exp/mel128hop1024/Cnn14_DecisionLevelAtt/${tag}/checkpoint-${step}"
    done
    for tag in v001 v003 v005; do
        rm -rf "exp/mel128hop1024/EfficientNet/${tag}/checkpoint-${step}"
    done
    for tag in v000 v002 v004; do
        rm -rf "exp/wave/EfficientNet/${tag}/checkpoint-${step}"
    done
done
du -s -h
