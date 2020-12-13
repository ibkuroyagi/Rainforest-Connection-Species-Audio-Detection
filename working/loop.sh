#!/bin/bash

# Copyright 2020 Ibuki Kuroyanagi.
# Created by Ibuki Kuroyanagi
sbatch ./run.sh \
    --tag "Cnn14_DecisionLevelAtt/base" \
    --stage "1" \
    --stop_stage "1" \
    --cal_type "0" \
    --verbose 1
