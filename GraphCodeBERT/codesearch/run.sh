pip install transformers tqdm tree_sitter > ~/log.txt 2>&1

lang=$1
mkdir -p ./saved_models/$lang
python run.py \
    --output_dir=./saved_models/$lang \
    --config_name=graphcodebert-base \
    --model_name_or_path=graphcodebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --lang=$lang \
    --do_train \
    --train_data_file=../dataset/$1/train.jsonl \
    --eval_data_file=../dataset/$1/valid.jsonl \
    --test_data_file=../dataset/$1/test.jsonl \
    --codebase_file=../dataset/$1/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --data_flow_length 64 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1| tee saved_models/$lang/train.log

python run.py \
    --output_dir=./saved_models/$lang \
    --config_name=graphcodebert-base \
    --model_name_or_path=graphcodebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --lang=$lang \
    --do_eval \
    --do_test \
    --train_data_file=../dataset/$1/train.jsonl \
    --eval_data_file=../dataset/$1/valid.jsonl \
    --test_data_file=../dataset/$1/test.jsonl \
    --codebase_file=../dataset/$1/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --data_flow_length 64 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1| tee saved_models/$lang/test.log
