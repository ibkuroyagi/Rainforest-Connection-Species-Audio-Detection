#!/bin/bash

# Copyright 2020 Ibuki Kuroyanagi.
# Created by Ibuki Kuroyanagi

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

. ./path.sh || exit 1
. ./cmd.sh || exit 1

# basic setting
stage=0        # stage to start
stop_stage=100 # stage to stop
n_gpus=1       # number of gpus for training
n_jobs=8       # number of parallel jobs in feature extraction
type=wave      # preprocess type.
cal_type=1     # if 1 -> statistic, else -> load cache pkl.
conf=conf/Cnn14_DecisionLevelAtt.yaml
verbose=1 # verbosity level, higher is more logging

# directory related
datadir="../input/rfcx-species-audio-detection"
dumpdir=dump
expdir=exp                        # directory to save experiments
tag="Cnn14_DecisionLevelAtt/base" # tag for manangement of the naming of experiments
cache_path="../input/pretrained/Cnn14_DecisionLevelAtt.pth"
resume=""
# evaluation related
checkpoint="best_score" # path of checkpoint to be used for evaluation
step="best"

. utils/parse_options.sh || exit 1

set -euo pipefail

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    log "Stage 0: Feature extraction."
    statistic_path="${dumpdir}/cache/${type}.pkl"
    [ ! -e "${dumpdir}/cache" ] && mkdir -p "${dumpdir}/cache"
    log "Feature extraction. See the progress via ${dumpdir}/preprocess.log"
    # shellcheck disable=SC2086
    ${train_cmd} --num_threads "${n_jobs}" "${dumpdir}/preprocess.log" \
        python ../input/modules/bin/preprocess.py \
        --datadir "${datadir}" \
        --dumpdir "${dumpdir}" \
        --config "${conf}" \
        --statistic_path "${statistic_path}" \
        --cal_type "${cal_type}" \
        --type "${type}" \
        --verbose "${verbose}"
    log "Successfully calculate logmel spectrogram."
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    log "Stage 1: Network training."
    outdir=${expdir}/${tag}
    log "Training start. See the progress via ${outdir}/sed_train.log"
    # shellcheck disable=SC2086
    ${cuda_cmd} --num_threads "${n_jobs}" --gpu "${n_gpus}" "${outdir}/sed_train.log" \
        python ../input/modules/bin/sed_train.py \
        --datadir "${datadir}" \
        --dumpdir "${dumpdir}/${type}" \
        --outdir "${outdir}" \
        --cache_path "${cache_path}" \
        --config "${conf}" \
        --resume "${resume}" \
        --verbose "${verbose}"
    log "Successfully finished training."
fi
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    log "Stage 2: Network inference."
    outdir=${expdir}/${tag}/${step}
    checkpoints=""
    for fold in {0..4}; do
        checkpoints+="${outdir}/${checkpoint}${fold}fold.pkl "
    done
    log "Inference start. See the progress via ${outdir}/inference.log"
    # shellcheck disable=SC2086
    ${cuda_cmd} --num_threads "${n_jobs}" --gpu "${n_gpus}" "${outdir}/inference.log" \
        python node_inference.py \
        --outdir "${outdir}" \
        --config "${conf}" \
        --checkpoints ${checkpoints} \
        --dpgmmdir "${dpgmmdir}" \
        --verbose "${verbose}"
    log "Successfully finished the inference."
fi
