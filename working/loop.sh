#!/bin/bash

# Copyright 2020 Ibuki Kuroyanagi.
# Created by Ibuki Kuroyanagi
mkdir conf/tuning
cp -a /home/i_kuroyanagi/workspace2/kaggle/moa/working/conf/slurm.conf conf/slurm.conf
cp -a -r /home/i_kuroyanagi/workspace2/kaggle/moa/working/utils utils
cp -a /home/i_kuroyanagi/workspace2/kaggle/moa/working/run.sh run.sh
cp -a /home/i_kuroyanagi/workspace2/kaggle/moa/working/path.sh path.sh
cp -a /home/i_kuroyanagi/workspace2/kaggle/moa/working/cmd.sh cmd.sh
echo END
