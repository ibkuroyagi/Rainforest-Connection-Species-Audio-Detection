#!/bin/bash

# Copyright 2020 Ibuki Kuroyanagi.
# Created by Ibuki Kuroyanagi
No=v002
model=Cnn14_DecisionLevelAtt
stage=1
stop_stage=1

sbatch -J "${model}/${No}" ./run.sh \
    --conf "conf/tuning/${model}.${No}.yaml" \
    --tag "${model}/${No}" \
    --stage "${stage}" \
    --stop_stage "${stop_stage}" \
    --cal_type "0" \
    --verbose 1
