#!/usr/bin/env bash

for dataset in 'MIntRec2.0'
do
    for seed in 0 1 2 3
    do
        python run.py \
        --dataset $dataset \
        --logger_name 'nmfir' \
        --method 'nmfir' \
        --data_mode 'multi-class' \
        --train \
        --save_results \
        --seed $seed \
        --gpu_id '1' \
        --video_feats_path 'video_feats.pkl' \
        --audio_feats_path 'audio_feats.pkl' \
        --text_backbone 'bert-large-uncased' \
        --config_file_name 'nmfir_bert_MIntRec2' \
        --results_file_name 'nmrfm_bert_MIntRec2_511.csv'
    done
done