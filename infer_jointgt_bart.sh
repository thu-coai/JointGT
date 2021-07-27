#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python cli_gt.py \
        --do_predict \
        --output_dir out/jointgt_bart_webnlg \
        --train_file data/webnlg/train \
        --predict_file data/webnlg/test \
        --tokenizer_path pretrain_model/jointgt_bart \
        --dataset webnlg \
        --predict_batch_size 32 \
        --max_input_length 256 \
        --max_output_length 128 \
        --append_another_bos \
        --num_beams 5 \
        --prefix test_beam5_lenpen1_

CUDA_VISIBLE_DEVICES=0 python cli_gt.py \
        --do_predict \
        --output_dir out/jointgt_bart_webnlg_const \
        --train_file data/webnlg_const/train \
        --predict_file data/webnlg_const/test \
        --tokenizer_path pretrain_model/jointgt_bart \
        --dataset webnlg \
        --predict_batch_size 32 \
        --max_input_length 256 \
        --max_output_length 128 \
        --append_another_bos \
        --num_beams 5 \
        --clean_up_spaces \
        --prefix test_beam5_lenpen1_

CUDA_VISIBLE_DEVICES=0 python cli_gt.py \
        --do_predict \
        --output_dir out/jointgt_bart_wq \
        --train_file data/wq/train \
        --predict_file data/wq/test \
        --tokenizer_path pretrain_model/jointgt_bart \
        --dataset webnlg \
        --predict_batch_size 32 \
        --max_input_length 256 \
        --max_output_length 128 \
        --append_another_bos \
        --num_beams 5 \
        --prefix test_beam5_lenpen5_ \
        --length_penalty 5.0

CUDA_VISIBLE_DEVICES=0 python cli_gt.py \
        --do_predict \
        --output_dir out/jointgt_bart_pq \
        --train_file data/pq/train \
        --predict_file data/pq/test \
        --tokenizer_path pretrain_model/jointgt_bart \
        --dataset webnlg \
        --predict_batch_size 32 \
        --max_input_length 128 \
        --max_output_length 64 \
        --append_another_bos \
        --num_beams 5 \
        --prefix test_beam5_lenpen1_
