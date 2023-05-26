

# Code Search

## Data Download

#### 1. AdvTest dataset

```bash
mkdir dataset && cd dataset
wget https://github.com/microsoft/CodeXGLUE/raw/main/Text-Code/NL-code-search-Adv/dataset.zip
unzip dataset.zip && rm -r dataset.zip && mv dataset AdvTest && cd AdvTest
wget https://zenodo.org/record/7857872/files/python.zip
unzip python.zip && python preprocess.py && rm -r python && rm -r *.pkl && rm python.zip
cd ../..
```

#### 2. CosQA dataset

```bash
cd dataset
mkdir cosqa && cd cosqa
wget https://github.com/Jun-jie-Huang/CoCLR/raw/main/data/search/code_idx_map.txt
wget https://github.com/Jun-jie-Huang/CoCLR/raw/main/data/search/cosqa-retrieval-dev-500.json
wget https://github.com/Jun-jie-Huang/CoCLR/raw/main/data/search/cosqa-retrieval-test-500.json
wget https://github.com/Jun-jie-Huang/CoCLR/raw/main/data/search/cosqa-retrieval-train-19604.json
cd ../..
```

#### 3. CSN dataset

```bash
cd dataset
wget https://github.com/microsoft/CodeBERT/raw/master/GraphCodeBERT/codesearch/dataset.zip
unzip dataset.zip && rm -r dataset.zip && mv dataset CSN && cd CSN
bash run.sh 
cd ../..
```



## Dependency 

- pip install torch
- pip install transformers

## Zero-Shot Setting

We first provide scripts for zero-shot code search. The similarity between code and nl we use is cosine distance of hidden states of UniXcoder.

#### 1. AdvTest dataset

```bash
python run.py \
    --output_dir saved_models/AdvTest \
    --model_name_or_path microsoft/unixcoder-base  \
    --do_zero_shot \
    --do_test \
    --test_data_file dataset/AdvTest/test.jsonl \
    --codebase_file dataset/AdvTest/test.jsonl \
    --num_train_epochs 2 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456
```

#### 2. CosQA dataset

```bash
python run.py \
    --output_dir saved_models/cosqa \
    --model_name_or_path microsoft/unixcoder-base  \
    --do_zero_shot \
    --do_test \
    --test_data_file dataset/cosqa/cosqa-retrieval-test-500.json \
    --codebase_file dataset/cosqa/code_idx_map.txt \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456
```

#### 3. CSN dataset

```bash
lang=python
python run.py \
    --output_dir saved_models/CSN/$lang \
    --model_name_or_path microsoft/unixcoder-base  \
    --do_zero_shot \
    --do_test \
    --test_data_file dataset/CSN/$lang/test.jsonl \
    --codebase_file dataset/CSN/$lang/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456
```



## Fine-Tune Setting

Here we provide fine-tune settings for code search, whose results are reported in the paper.

#### 1. AdvTest dataset

```shell
# Training
python run.py \
    --output_dir saved_models/AdvTest \
    --model_name_or_path microsoft/unixcoder-base  \
    --do_train \
    --train_data_file dataset/AdvTest/train.jsonl \
    --eval_data_file dataset/AdvTest/valid.jsonl \
    --codebase_file dataset/AdvTest/valid.jsonl \
    --num_train_epochs 2 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456
    
# Evaluating
python run.py \
    --output_dir saved_models/AdvTest \
    --model_name_or_path microsoft/unixcoder-base  \
    --do_test \
    --test_data_file dataset/AdvTest/test.jsonl \
    --codebase_file dataset/AdvTest/test.jsonl \
    --num_train_epochs 2 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456
```
#### 2. CosQA dataset

```bash
# Training
python run.py \
    --output_dir saved_models/cosqa \
    --model_name_or_path microsoft/unixcoder-base  \
    --do_train \
    --train_data_file dataset/cosqa/cosqa-retrieval-train-19604.json \
    --eval_data_file dataset/cosqa/cosqa-retrieval-dev-500.json \
    --codebase_file dataset/cosqa/code_idx_map.txt \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456

# Evaluating
python run.py \
    --output_dir saved_models/cosqa \
    --model_name_or_path microsoft/unixcoder-base  \
    --do_eval \
    --do_test \
    --eval_data_file dataset/cosqa/cosqa-retrieval-dev-500.json \
    --test_data_file dataset/cosqa/cosqa-retrieval-test-500.json \
    --codebase_file dataset/cosqa/code_idx_map.txt \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456 
```

#### 3. CSN dataset

```bash
# Training
lang=python
python run.py \
    --output_dir saved_models/CSN/$lang \
    --model_name_or_path microsoft/unixcoder-base  \
    --do_train \
    --train_data_file dataset/CSN/$lang/train.jsonl \
    --eval_data_file dataset/CSN/$lang/valid.jsonl \
    --codebase_file dataset/CSN/$lang/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456 

# Evaluating
python run.py \
    --output_dir saved_models/CSN/$lang \
    --model_name_or_path microsoft/unixcoder-base  \
    --do_eval \
    --do_test \
    --eval_data_file dataset/CSN/$lang/valid.jsonl \
    --test_data_file dataset/CSN/$lang/test.jsonl \
    --codebase_file dataset/CSN/$lang/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456

```

