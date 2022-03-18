#!/usr/bin/env bash

set -e

while getopts "ln:" opt; do
      case $opt in
        l ) LABELS="true";;
        n ) NSPEAKERS=$OPTARG;;
        \?) echo "Invalid option: -"$OPTARG"" >&2
            exit 1;;
      esac
    done

: ${NSPEAKERS:=1}  # default value
: ${DATA_DIR:=LJSpeech-1.1}
: ${WAV_DIR:=${DATA_DIR}/wavs}  # should already exist
: ${FILELIST:=filelists/ljs_audio_text.txt}
: ${ALIGNMENT_DIR:=${DATA_DIR}/mfa_alignments}
: ${ARGS="--extract-mels"}

if [ "$LABELS" = "true" ]
then
  python ./create_lab_files.py --dataset ${WAV_DIR} --filelist ${FILELIST} --n-speakers ${NSPEAKERS}
fi

#mfa model download acoustic english --temp_directory /disk/scratch1/evdv/tmp/MFA
#mfa model download dictionary english  --temp_directory /disk/scratch1/evdv/tmp/MFA
#mfa validate ${WAV_DIR} english english  --temp_directory /disk/scratch1/evdv/tmp/MFA
#mfa align ${WAV_DIR} english english ${ALIGNMENT_DIR}  --temp_directory /disk/scratch1/evdv/tmp/MFA

# don't change batch size
python prepare_dataset.py \
    --wav-text-filelists ${FILELIST} \
    --n-workers 4 \
    --batch-size 1 \
    --dataset-path $DATA_DIR \
    --textgrid-path $ALIGNMENT_DIR \
    --extract-pitch \
    --extract-durations\
    --durs-online-dir "/tmp/" \
    --f0-method pyin \
    $ARGS
