#!/usr/bin/env bash

## install miniconda:
while getopts "cu:" opt; do
      case $opt in
        c ) CONDA="true";;
        u ) USER="$OPTARG";;
        \?) echo "Invalid option: -"$OPTARG"" >&2
            exit 1;;
      esac
    done
: ${CONDA-"false"}  # default value
: ${USER-`whoami`}  # default value

if [ "$CONDA" = "true" ]
then
  cd /disk/scratch1/${USER}/
  wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh
  echo "!! Change install location to /disk/scratch1/${USER}/miniconda3 !!"
  bash ./Miniconda3-py39_4.9.2-Linux-x86_64.sh
  rm Miniconda3-py39_4.9.2-Linux-x86_64.sh*
  source ~/.bashrc
fi

source /disk/scratch1/${USER}/miniconda3/bin/activate

SERVERNAME=`hostname -s`
conda create -n fastpitch_${SERVERNAME} python=3.8

## Get a version of gcc > 5.0 which works to compile apex
conda install -n fastpitch_${SERVERNAME} gcc_linux-64=8.4 gxx_linux-64=8.4
## activating the environment now sets CC and CXX environment variables
## to point to our gcc/g++
source activate fastpitch_${SERVERNAME}

export CUDA_HOME=/opt/cuda-10.2.89_440_33
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

## Apex
cd /disk/scratch1/${USER}/FastPitches/PyTorch/SpeechSynthesis/FastPitch/
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../

## Python requirements
## ignore warnings around numba installation
pip install -r requirements.txt

## for logging
## if needed, create a free account here: https://app.wandb.ai/login?signup=true
wandb login

export CUDA_VISIBLE_DEVICES=1


## Test installation
./scripts/download_fastpitch.sh
./scripts/download_waveglow.sh
mkdir output
python inference.py --cuda --fastpitch pretrained_models/fastpitch/nvidia_fastpitch_210824.pt --waveglow pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt --wn-channels 256 -i phrases/devset10.tsv -o output/wavs_devset10


## Get set up with LJ
./scripts/download_dataset.sh
./scripts/prepare_dataset.sh
