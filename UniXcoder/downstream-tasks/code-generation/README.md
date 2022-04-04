# Code Generation

## Data Download

```bash
mkdir dataset
cd dataset
wget https://github.com/microsoft/CodeXGLUE/raw/main/Text-Code/text-to-code/dataset/concode/train.json
wget https://github.com/microsoft/CodeXGLUE/raw/main/Text-Code/text-to-code/dataset/concode/dev.json
wget https://github.com/microsoft/CodeXGLUE/raw/main/Text-Code/text-to-code/dataset/concode/test.json
cd ..
```

## Dependency 

- pip install torch
- pip install transformers

## Fine-Tune Setting

Here we provide fine-tune settings for code generation, whose results are reported in the paper.

```shell
# Training
python run.py \
	--do_train \
	--do_eval \
	--model_name_or_path microsoft/unixcoder-base \
	--train_filename dataset/train.json \
	--dev_filename dataset/dev.json \
	--output_dir saved_models \
	--max_source_length 350 \
	--max_target_length 150 \
	--beam_size 3 \
	--train_batch_size 32 \
	--eval_batch_size 32 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 1 \
	--num_train_epochs 30 

# Output results
python run.py \
	--do_test \
	--model_name_or_path microsoft/unixcoder-base \
	--test_filename dataset/test.json \
	--output_dir saved_models \
	--max_source_length 350 \
	--max_target_length 150 \
	--beam_size 3 \
	--train_batch_size 32 \
	--eval_batch_size 32 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 1 \
	--num_train_epochs 30 
```

Prediction results of test set are  ```saved_models/predictions.txt```.To obtain the score of test set, you need to send the prediction to codexglue@microsoft.com.

