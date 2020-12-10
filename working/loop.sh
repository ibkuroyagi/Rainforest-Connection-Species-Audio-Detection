#!/bin/bash

# Copyright 2020 Ibuki Kuroyanagi.
# Created by Ibuki Kuroyanagi
sbatch -c 4 ./run.sh \
    --tag "base" \
    --stop_stage "0"
