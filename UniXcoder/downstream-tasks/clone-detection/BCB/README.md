# Clone Detection (BigCloneDetection)

## Data Download

```bash
mkdir dataset
cd dataset
wget https://github.com/microsoft/CodeXGLUE/raw/main/Code-Code/Clone-detection-BigCloneBench/dataset/data.jsonl
wget https://github.com/microsoft/CodeXGLUE/raw/main/Code-Code/Clone-detection-BigCloneBench/dataset/test.txt
wget https://github.com/microsoft/CodeXGLUE/raw/main/Code-Code/Clone-detection-BigCloneBench/dataset/train.txt
wget https://github.com/microsoft/CodeXGLUE/raw/main/Code-Code/Clone-detection-BigCloneBench/dataset/valid.txt
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
    --train_data_file dataset/train.txt \
    --eval_data_file dataset/valid.txt \
    --num_train_epochs 1 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --seed 123456 
    
# Evaluating
python run.py \
    --output_dir saved_models \
    --model_name_or_path microsoft/unixcoder-base \
    --do_test \
    --test_data_file dataset/test.txt \
    --num_train_epochs 1 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --seed 123456 
```
