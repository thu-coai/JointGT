#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2 python cli_gt.py \
        --do_train \
        --do_pretrain \
        --model_path pretrain_model/bart \
        --tokenizer_path pretrain_model/bart \
        --output_dir pretrain_model/jointgt_bart \
        --train_file kgpt/dataset/wikidata/train \
        --predict_file kgpt/dataset/wikidata/val \
        --knowledge_file kgpt/preprocess/knowledge-full \
        --train_batch_size 42 \
        --predict_batch_size 42 \
        --max_input_length 600 \
        --max_output_length 64 \
        --gradient_accumulation_steps 4 \
        --append_another_bos \
        --learning_rate 3e-5 \
        --num_train_epochs 1 \
        --warmup_steps 4000 \
        --eval_period 2000 \
        --save_period 35000 \
        --num_workers 4
