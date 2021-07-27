#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python cli_gt.py \
        --do_train \
        --model_name t5 \
        --output_dir out/jointgt_t5_webnlg \
        --train_file data/webnlg/train \
        --predict_file data/webnlg/val \
        --model_path pretrain_model/jointgt_t5 \
        --tokenizer_path pretrain_model/jointgt_t5 \
        --dataset webnlg \
        --train_batch_size 24 \
        --predict_batch_size 24 \
        --max_input_length 256 \
        --max_output_length 128 \
        --append_another_bos \
        --learning_rate 5e-5 \
        --num_train_epochs 30 \
        --warmup_steps 1600 \
        --eval_period 800 \
        --num_beams 5

CUDA_VISIBLE_DEVICES=0,1 python cli_gt.py \
        --do_train \
        --model_name t5 \
        --output_dir out/jointgt_t5_webnlg_const \
        --train_file data/webnlg_const/train \
        --predict_file data/webnlg_const/dev \
        --model_path pretrain_model/jointgt_t5 \
        --tokenizer_path pretrain_model/jointgt_t5 \
        --dataset webnlg \
        --train_batch_size 32 \
        --predict_batch_size 32 \
        --max_input_length 256 \
        --max_output_length 128 \
        --append_another_bos \
        --learning_rate 3e-5 \
        --num_train_epochs 30 \
        --warmup_steps 1200 \
        --eval_period 600 \
        --num_beams 5 \
        --clean_up_spaces

CUDA_VISIBLE_DEVICES=0,1 python cli_gt.py \
        --do_train \
        --model_name t5 \
        --output_dir out/jointgt_t5_wq \
        --train_file data/wq/train \
        --predict_file data/wq/dev \
        --model_path pretrain_model/jointgt_t5 \
        --tokenizer_path pretrain_model/jointgt_t5 \
        --dataset webnlg \
        --train_batch_size 32 \
        --predict_batch_size 32 \
        --max_input_length 256 \
        --max_output_length 64 \
        --learning_rate 1e-4 \
        --num_train_epochs 40 \
        --warmup_steps 2300 \
        --eval_period 600 \
        --num_beams 5

CUDA_VISIBLE_DEVICES=0,1 python -u cli_gt.py \
        --do_train \
        --model_name t5 \
        --output_dir out/jointgt_t5_pq \
        --train_file data/pq/train \
        --predict_file data/pq/dev \
        --model_path pretrain_model/jointgt_t5 \
        --tokenizer_path pretrain_model/jointgt_t5 \
        --dataset webnlg \
        --train_batch_size 32 \
        --predict_batch_size 32 \
        --max_input_length 128 \
        --max_output_length 64 \
        --learning_rate 2e-5 \
        --num_train_epochs 30 \
        --warmup_steps 900 \
        --eval_period 300 \
        --num_beams 5
