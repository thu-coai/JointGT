# JointGT: Graph-Text Joint Representation Learning for Text Generation from Knowledge Graphs

## Introduction

JointGT is a graph-text joint pre-training framework with structure-aware encoding and explicit graph-text alignments. You can read our [paper](https://aclanthology.org/2021.findings-acl.223/) for more details. This project is a PyTorch implementation of our work.

## Dependencies

* Python 3.7
* NumPy
* PyTorch 1.4.0
* Transformers (Huggingface) 3.0.0
* [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter) 2.0.4

## Quick Start for Fine-tuning

**NOTE**: At the very beginning, in order to compute the METEOR scores, please download the required [data](https://github.com/xinyadu/nqg/blob/master/qgevalcap/meteor/data/paraphrase-en.gz) and put it under the following two folders: `eval_webnlg/pycocoevalcap/meteor/data/` and `eval_wqpq/meteor/data/`.

### Datasets

Our experiments contain four downstream datasets, i.e., WebNLG(U), WebNLG(C), WebQuestions, and PathQuestions. The raw data of these datasets are from the GitHub repositories of [KGPT](https://github.com/wenhuchen/KGPT), [WebNLG](https://gitlab.com/shimorina/webnlg-dataset/), and [BiGGNN](https://github.com/hugochan/Graph2Seq-for-KGQG). You can download the pre-processed datasets used in our paper on [Tsinghua Cloud]().

### Fine-tuning

You can download the checkpoint of our pre-trained model ([Tsinghua Cloud]()), and fine-tune the pre-trained model on four datasets.

```shell
bash finetune_jointgt_bart.sh
bash finetune_jointgt_t5.sh
```

In the scripts, `--output_dir` denotes the directory to save the fine-tuning model. `--model_path` indicates the pre-trained checkpoint used for fine-tuning. You can refer to the fine-tuning codes for the description of other hyper-parameters.

### Inference

We also provide the inference scripts to directly acquire the generation results on the test sets.

```shell
bash infer_jointgt_bart.sh
bash infer_jointgt_t5.sh
```

In the scripts, `--output_dir` denotes the directory of model checkpoint used for inference. The generated results are also saved in this directory.

## Pre-training

If you want to conduct pre-training by yourself instead of directly using the checkpoint we provide, you should first download the KGTEXT dataset and the corresponding knowledge graphs from the GitHub repository of [KGPT](https://github.com/wenhuchen/KGPT). Then, the model checkpoint of [BART](https://huggingface.co/facebook/bart-base) / [T5](https://huggingface.co/t5-base) provided by Huggingface Transformers should also be prepared as the initialization of our model.

We provide the scripts for pre-training as follows.

```shell
bash pretrain_jointgt_bart.sh
bash pretrain_jointgt_t5.sh
```

In the scripts, `--model_path` and `--tokenizer_path` are set to the downloaded BART / T5 checkpoint. The settings of`--train_file`, `--predict_file` and `--knowledge_file` depend on the path of datasets and knowledge graphs from KGPT.

## Evaluation

For a fair comparison with existing works, we use the evaluation scripts of [KGPT](https://github.com/wenhuchen/KGPT) for WebNLG.

```shell
cd eval_webnlg
python measure_score.py ${reference_path} ${model_output_path}
```

As for WebQuestions and PathQuestions, we use the scripts of [BiGGNN](https://github.com/hugochan/Graph2Seq-for-KGQG) for evaluation.

```shell
cd eval_for_wqpq
python eval.py --src ${source_path} --tgt ${reference_path} --out ${model_output_path}
```

During evaluation, `model_output_path` can be set to the generated file when running our inference codes. `source_path` can be set to `test.source` / `src-test.txt` in our pre-processed datasets. `reference_path` can be set to `test.target` / `tgt-test.txt` in our pre-processed datasets. More details can refer to the original repositories.

## Citation

```
@inproceedings{ke-etal-2021-jointgt,
    title = "{J}oint{GT}: Graph-Text Joint Representation Learning for Text Generation from Knowledge Graphs",
    author = "Ke, Pei  and Ji, Haozhe and Ran, Yu and Cui, Xin and Wang, Liwei and Song, Linfeng and Zhu, Xiaoyan and Huang, Minlie",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    pages = "2526--2538",
}
```

Please kindly cite our paper if this paper and the codes are helpful.

## Thanks

Many thanks to the GitHub repositories of [Transformers](https://github.com/huggingface/transformers), [bart-closed-book-qa](https://github.com/shmsw25/bart-closed-book-qa) and [KGPT](https://github.com/wenhuchen/KGPT). Part of our codes are modified based on their codes.
