#!/bin/bash

# Copyright 2020 Ibuki Kuroyanagi.
# Created by Ibuki Kuroyanagi
sbatch ./run.sh \
    --tag "base" \
    --stop_stage "0" \
    --cal_type "0"
