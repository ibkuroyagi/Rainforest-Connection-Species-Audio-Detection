sbatch -c 4 -J preprocess1.1 <<EOF
#!/bin/sh
. ./path.sh || exit 1
. ./cmd.sh || exit 1
python ../input/modules/bin/preprocess.py --datadir ../input/rfcx-species-audio-detection --dumpdir dump --config conf/tuning/EfficientNet.v008.yaml --statistic_path dump/cache/wave.pkl --cal_type 0 --type wave_sp1.1 --facter 1.1 --verbose 1
EOF
