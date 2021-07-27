#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python cli_gt.py \
        --do_predict \
        --model_name t5 \
        --output_dir out/jointgt_t5_webnlg \
        --train_file data/webnlg/train \
        --predict_file data/webnlg/test \
        --tokenizer_path pretrain_model/jointgt_t5 \
        --dataset webnlg \
        --predict_batch_size 24 \
        --max_input_length 256 \
        --max_output_length 128 \
        --num_beams 5 \
        --prefix test_beam5_lenpen1_

CUDA_VISIBLE_DEVICES=0 python cli_gt.py \
        --do_predict \
        --model_name t5 \
        --output_dir out/jointgt_t5_webnlg_const \
        --train_file data/webnlg_const/train \
        --predict_file data/webnlg_const/test \
        --tokenizer_path pretrain_model/jointgt_t5 \
        --dataset webnlg \
        --predict_batch_size 32 \
        --max_input_length 256 \
        --max_output_length 128 \
        --num_beams 5 \
        --clean_up_spaces \
        --prefix test_beam5_lenpen1_

CUDA_VISIBLE_DEVICES=0 python cli_gt.py \
        --do_predict \
        --model_name t5 \
        --output_dir out/jointgt_t5_wq \
        --train_file data/wq/train \
        --predict_file data/wq/test \
        --tokenizer_path pretrain_model/jointgt_t5 \
        --dataset webnlg \
        --predict_batch_size 32 \
        --max_input_length 256 \
        --max_output_length 64 \
        --num_beams 5 \
        --length_penalty 5.0 \
        --prefix test_beam5_lenpen5_

CUDA_VISIBLE_DEVICES=0 python cli_gt.py \
        --do_predict \
        --model_name t5 \
        --output_dir out/jointgt_t5_pq \
        --train_file data/pq/train \
        --predict_file data/pq/test \
        --tokenizer_path pretrain_model/jointgt_t5 \
        --dataset webnlg \
        --predict_batch_size 32 \
        --max_input_length 128 \
        --max_output_length 64 \
        --num_beams 2 \
        --length_penalty 1.0 \
        --prefix test_beam2_lenpen1_
