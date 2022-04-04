# Clone Detection (POJ-104)

## Data Download

```bash
cd dataset
pip install gdown
gdown https://drive.google.com/uc?id=0B2i-vWnOu7MxVlJwQXN6eVNONUU
tar -xvf programs.tar.gz
python preprocess.py
cd ..
```

## Dependency 

- pip install torch
- pip install transformers

## Fine-Tune

Here we provide fine-tune settings for code summarization, whose results are reported in the paper.

```shell
# Training
python run.py \
    --output_dir saved_models \
    --model_name_or_path microsoft/unixcoder-base \
    --do_train \
    --train_data_file dataset/train.jsonl \
    --eval_data_file dataset/valid.jsonl \
    --test_data_file dataset/test.jsonl \
    --num_train_epochs 2 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 123456
    
# Evaluating	
python run.py \
    --output_dir saved_models \
    --model_name_or_path microsoft/unixcoder-base \
    --do_eval \
    --do_test \
    --eval_data_file dataset/valid.jsonl \
    --test_data_file dataset/test.jsonl \
    --num_train_epochs 2 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 123456
```
