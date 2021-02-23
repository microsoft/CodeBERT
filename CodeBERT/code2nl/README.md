# Code Documentation Generation

This repo provides the code for reproducing the experiments on [CodeSearchNet](https://arxiv.org/abs/1909.09436) dataset for code document generation tasks in six programming languages.

**!News: We release a new pipeline for this task. The new pipeline only needs 2 p100 GPUs and less training time for Code Documentation Generation. Please refer to the [website](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text).**

## Dependency

- pip install torch==1.4.0
- pip install transformers==2.5.0
- pip install filelock

## Data Preprocess

We clean CodeSearchNet dataset for this task by following steps:

- Remove comments in the code
- Remove examples that codes cannot be parsed into an abstract syntax tree.
- Remove examples that #tokens of documents is < 3 or >256
- Remove examples that documents contain special tokens (e.g. <img ...> or https:...)
- Remove examples that documents are not English.

Data statistic about the cleaned dataset for code document generation is shown in this Table. We release the cleaned dataset in this [website](https://drive.google.com/open?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h).

| PL         | Training |  Dev   |  Test  |
| :--------- | :------: | :----: | :----: |
| Python     | 251,820  | 13,914 | 14,918 |
| PHP        | 241,241  | 12,982 | 14,014 |
| Go         | 167,288  | 7,325  | 8,122  |
| Java       | 164,923  | 5,183  | 10,955 |
| JavaScript |  58,025  | 3,885  | 3,291  |
| Ruby       |  24,927  | 1,400  | 1,261  |



## Data Download

You can download dataset from the [website](https://drive.google.com/open?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h). Or use the following command.

```shell
pip install gdown
mkdir data data/code2nl
cd data/code2nl
gdown https://drive.google.com/uc?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h
unzip Cleaned_CodeSearchNet.zip
rm Cleaned_CodeSearchNet.zip
cd ../..
```



## Fine-Tune

We fine-tuned the model on 4*P40 GPUs. 

```shell
cd code2nl

lang=php #programming language
lr=5e-5
batch_size=64
beam_size=10
source_length=256
target_length=128
data_dir=../data/code2nl/CodeSearchNet
output_dir=model/$lang
train_file=$data_dir/$lang/train.jsonl
dev_file=$data_dir/$lang/valid.jsonl
eval_steps=1000 #400 for ruby, 600 for javascript, 1000 for others
train_steps=50000 #20000 for ruby, 30000 for javascript, 50000 for others
pretrained_model=microsoft/codebert-base #Roberta: roberta-base

python run.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --train_steps $train_steps --eval_steps $eval_steps 
```



## Inference and Evaluation

After fine-tuning, inference and evaluation are as follows:

```shell
lang=php #programming language
beam_size=10
batch_size=128
source_length=256
target_length=128
output_dir=model/$lang
data_dir=../data/code2nl/CodeSearchNet
dev_file=$data_dir/$lang/valid.jsonl
test_file=$data_dir/$lang/test.jsonl
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

python run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size
```

The results on CodeSearchNet are shown in this Table:

| Model       |   Ruby    | Javascript |    Go     |  Python   |   Java    |    PHP    |  Overall  |
| ----------- | :-------: | :--------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| Seq2Seq     |   9.64    |   10.21    |   13.98   |   15.93   |   15.09   |   21.08   |   14.32   |
| Transformer |   11.18   |   11.59    |   16.38   |   15.81   |   16.26   |   22.12   |   15.56   |
| RoBERTa     |   11.17   |   11.90    |   17.72   |   18.14   |   16.47   |   24.02   |   16.57   |
| CodeBERT    | **12.16** | **14.90**  | **18.07** | **19.06** | **17.65** | **25.16** | **17.83** |


