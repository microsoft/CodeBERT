# Code Summarization

## Data Download

```bash
wget https://github.com/microsoft/CodeXGLUE/raw/main/Code-Text/code-to-text/dataset.zip
unzip dataset.zip
rm dataset.zip
cd dataset
wget https://zenodo.org/record/7857872/files/python.zip
wget https://zenodo.org/record/7857872/files/java.zip
wget https://zenodo.org/record/7857872/files/ruby.zip
wget https://zenodo.org/record/7857872/files/javascript.zip
wget https://zenodo.org/record/7857872/files/go.zip
wget https://zenodo.org/record/7857872/files/php.zip

unzip python.zip
unzip java.zip
unzip ruby.zip
unzip javascript.zip
unzip go.zip
unzip php.zip
rm *.zip
rm *.pkl

python preprocess.py
rm -r */final
cd ..
```

## Dependency 

- pip install torch
- pip install transformers

## Fine-Tune Setting

Here we provide fine-tune settings for code summarization, whose results are reported in the paper.

```shell
lang=python

# Training
python run.py \
	--do_train \
	--do_eval \
	--model_name_or_path microsoft/unixcoder-base \
	--train_filename dataset/$lang/train.jsonl \
	--dev_filename dataset/$lang/valid.jsonl \
	--output_dir saved_models/$lang \
	--max_source_length 256 \
	--max_target_length 128 \
	--beam_size 10 \
	--train_batch_size 48 \
	--eval_batch_size 48 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 2 \
	--num_train_epochs 10 
	
# Evaluating	
python run.py \
	--do_test \
	--model_name_or_path microsoft/unixcoder-base \
	--test_filename dataset/$lang/test.jsonl \
	--output_dir saved_models/$lang \
	--max_source_length 256 \
	--max_target_length 128 \
	--beam_size 10 \
	--train_batch_size 48 \
	--eval_batch_size 48 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 2 \
	--num_train_epochs 10 	
```
