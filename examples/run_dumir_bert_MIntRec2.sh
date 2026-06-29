#!/usr/bin/env bash

for dataset in 'MIntRec2.0'
do
for seed in  3
    do
        python run.py \
        --dataset $dataset \
        --logger_name 'dumir' \
        --method 'dumir' \
        --data_mode 'multi-class' \
        --train \
        --save_results \
        --seed $seed \
        --gpu_id '4' \
        --video_feats_path 'video_feats.pkl' \
        --audio_feats_path 'audio_feats.pkl' \
        --text_backbone 'bert-large-uncased' \
        --config_file_name 'dumir_bert_MIntRec2' \
        --results_file_name 'dumir_bert_MIntRec2.csv'
    done
done