#!/usr/bin/env bash

for dataset in 'MIntRec'
do
    for seed in 3
    do
        python run.py \
        --dataset $dataset \
        --logger_name 'nmfir' \
        --method 'nmfir' \
        --data_mode 'multi-class' \
        --train \
        --save_results \
        --seed $seed \
        --gpu_id '2' \
        --video_feats_path 'video_feats.pkl' \
        --audio_feats_path 'audio_feats.pkl' \
        --text_backbone 'bert-large-uncased' \
        --config_file_name 'nmfir_bert_MIntRec' \
        --results_file_name 'nmfir_bert_MIntRec_maa.csv'
    done
done