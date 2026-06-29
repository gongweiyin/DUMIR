#!/usr/bin/env bash

for dataset in 'MIntRec2.0'
do
    for seed in 0 1 2 3
    do
        python run.py \
        --dataset $dataset \
        --logger_name 'mag_bert' \
        --method 'mag_bert' \
        --data_mode 'multi-class' \
        --train \
        --save_results \
        --seed $seed \
        --gpu_id '1' \
        --video_feats_path 'video_feats.pkl' \
        --audio_feats_path 'audio_feats.pkl' \
        --text_backbone 'bert-large-uncased' \
        --config_file_name 'mag_bert_MIntRec2' \
        --results_file_name 'mag_bert_MIntRec2_ddl.csv'
    done
done