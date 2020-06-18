# CodeBERT
This repo provides the codes and model of the work [CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/pdf/2002.08155.pdf). The model is pre-trained on function-docstring data from CodeSearchNet in 6 programming languages, including Python, Java, Javascript, Ruby, PHP, and Go.

### Dependency

- pip install torch==1.4.0
- pip install transformers==2.5.0
- pip install filelock
  
## Pre-trained Model

We have released the pre-trained model as described in the paper.

You can download the pre-trained model (CodeBERT) from the [website](https://drive.google.com/open?id=1Rw60M7A1h4L_nHfeLRhi7Z8H3EHvuHRG). Or use the following command.

```shell
pip install gdown
mkdir pretrained_models
gdown https://drive.google.com/uc?id=1Rw60M7A1h4L_nHfeLRhi7Z8H3EHvuHRG
unzip pretrained_codebert.zip
rm  pretrained_codebert.zip
cd ..
```
### Qiuck Tour
We use huggingface/transformers framework to train the model. You can use our model like the pre-trained Roberta base. Now, We give an example on how to load the model.
```python
import argparse
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

config_path = "./pretrained_models/config.json"
model_path = "./pretrained_models/pytorch_model.bin"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = RobertaConfig.from_pretrained(config_path)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained(model_path, from_tf=False, config=config)
model.to(device)
```

## Code Search

### Data Preprocess

Both training and validation datasets are created in a way that positive and negative samples are balanced. Negative samples consist of balanced number of instances with randomly replaced NL and PL.

We follow the official evaluation metric to calculate the Mean Reciprocal Rank (MRR) for each pair of test data (c, w) over a fixed set of 999 distractor codes.

You can use the following command to download the preprocessed training and validation dataset and preprocess the test dataset by yourself. The preprocessed testing dataset is very large, so only the preprocessing script is provided.

```shell
mkdir data data/codesearch
cd data/codesearch
gdown https://drive.google.com/uc?id=1xgSR34XO8xXZg4cZScDYj2eGerBE9iGo  
unzip codesearch_data.zip
rm  codesearch_data.zip
cd ../../codesearch
python process_data.py
cd ..
```

### Fine-Tune
We fine-tuned the model on 2*P100 GPUs. 
```shell
cd codesearch

lang=php #fine-tuning a language-specific model for each programming language 
pretrained_model=../pretrained_models/pytorch_model.bin  #CodeBERT: path to .bin file. Roberta: roberta-base

python run_classifier.py \
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file train.txt \
--dev_file valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ../data/codesearch/train_valid/$lang \
--output_dir ./models/$lang  \
--model_name_or_path $pretrained_model \
--config_name roberta-base
```
### Inference and Evaluation

Inference
```shell
lang=php #programming language
idx=0 #test batch idx

python run_classifier.py \
--model_type roberta \
--model_name_or_path roberta-base \
--task_name codesearch \
--do_predict \
--output_dir ../data/codesearch/test/$lang \
--data_dir ../data/codesearch/test/$lang \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--test_file batch_${idx}.txt \
--pred_model_dir ./models/$lang/checkpoint-best/pytorch-model.bin \
--test_result_dir ./results/$lang/${idx}_batch_result.txt
```

Evaluation
```shell
python mrr.py
```

### Contact
Feel free to contact Daya Guo (guody5@mail2.sysu.edu.cn) and Duyu Tang (dutang@microsoft.com) if you have any further questions.
