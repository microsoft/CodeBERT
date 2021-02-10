# Code Search

## Data Preprocess

Different from the setting of [CodeSearchNet](husain2019codesearchnet), the answer of each query is retrieved from the whole development and testing code corpus instead of 1,000 candidate codes. Besides, we observe that some queries contain content unrelated to the code, such as a link ``http://..." that refers to external resources.  Therefore, we filter following examples to improve the quality of the dataset. 

- Remove comments in the code
- Remove examples that codes cannot be parsed into an abstract syntax tree.
- Remove examples that #tokens of documents is < 3 or >256
- Remove examples that documents contain special tokens (e.g. <img ...> or https:...)
- Remove examples that documents are not English.

```shell
unzip dataset.zip
cd dataset
bash run.sh 
cd ..
```

## Dependency 

- pip install torch
- pip install transformers
- pip install tree_sitter

## Fine-Tune

We fine-tuned the model on 2*P100 GPUs. 
```shell
lang=ruby
mkdir -p ./saved_models/$lang
python run.py \
    --output_dir=./saved_models/$lang \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --lang=$lang \
    --do_train \
    --train_data_file=dataset/$1/train.jsonl \
    --eval_data_file=dataset/$1/valid.jsonl \
    --test_data_file=dataset/$1/test.jsonl \
    --codebase_file=dataset/$1/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --data_flow_length 64 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1| tee saved_models/$lang/train.log
```
## Inference and Evaluation

```shell
lang=ruby
python run.py \
    --output_dir=./saved_models/$lang \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --lang=$lang \
    --do_eval \
    --do_test \
    --train_data_file=dataset/$1/train.jsonl \
    --eval_data_file=dataset/$1/valid.jsonl \
    --test_data_file=dataset/$1/test.jsonl \
    --codebase_file=dataset/$1/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --data_flow_length 64 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1| tee saved_models/$lang/test.log
```


