# CodeExecutor

This repo provides the code for reproducing the experiments in [Code Execution with Pre-trained Language Models](https://arxiv.org/pdf/2305.05383.pdf). **CodeExecutor** is a pre-trained model that learns to predict the execution traces using a code execution pre-training task and curriculum learning.

The pre-trained checkpoint of CodeExecutor is available on [Huggingface](https://huggingface.co/microsoft/codeexecutor). 

Our dataset is available on [Zenodo](https://zenodo.org/record/8062703).

## 1. Dependency

- pip install pytorch 
- pip install transformers
- pip install python-Levenshtein


## 2. Data

The **Python Code Execution datasets** are a series of datasets following an easy-to-hard paradigm, including the **SingleLine** **dataset**, **Tutorial** **dataset**, and **CodeNetMut** **dataset**. We provide each test set of the three on [Zenodo](https://zenodo.org/record/8062703).

Demo data (simplified version):

``` python
{
    "id": 0,  
    "code": "s = ['x', 'y', 'z']",  
    "code_tokens": ["<0>", "s", "=", "[", "'x'", ",", "'y'", ",", "'z'", "]"],  
    "trace": ["<line> <0> <state> s : [ x , y , z ] </state>"],
    "trace_tokens": ["<line>", "<0>", "<state>", "s", ":", "[", "x", ",", "y", ",", "z", "]", "</state>"]

}
```

We also construct a new dataset for the **zero-shot code-to-code search task**, by collecting 9,987 Python functions from CodeNet. Each function solves one of the 48 problems. 

Demo data (simplified version):

``` python
{
    "id": 0,  
    "code_id": "s204511158", 
    "problem_id": 340, # solve which problem
    "original_code": "s = list(input())", # code without providing the test case
    "code": "s = ['x', 'y', 'z']",  # code provided with a test case
    "code_tokens": ["<0>", "s", "=", "[", "'x'", ",", "'y'", ",", "'z'", "]"],  
    "trace": ["<line> <0> <state> s : [ x , y , z ] </state>"],
    "trace_tokens": ["<line>", "<0>", "<state>", "s", ":", "[", "x", ",", "y", ",", "z", "]", "</state>"]
}
```


## 3. Pre-training

```bash
# prepare model checkpoint and datasets
cd pretrain
bash run.sh
```

A demo bash script (run.sh) is shown:
```bash
# Change the arguments as required:
#   output_dir: the output directory to save inference results
#   data_cache_dir: the output directory to save the data cache 
#   train_data_path: the path of the pre-training file
#   eval_data_path: the path of the test file
#   model_name_or_path: the path of the model to be evaluated

PER_NODE_GPU=8
python -m torch.distributed.launch --nproc_per_node=${PER_NODE_GPU} run.py \
    --output_dir ../saved_models/pretrain_codeexecutor_stage_3 \
    --data_cache_dir ../saved_models/pretrain_codeexecutor_stage_3 \
    --train_data_path /drive/pretrain_codenetmut.json \
    --another_train_data_path /drive/pretrain_tutorial.json \
    --third_train_data_path /drive/single_line_hard_3_million.json \
    --eval_data_path ../data/codenetmut_test.json \
    --model_name_or_path ../saved_models/pretrain_codeexecutor_stage_2 \
    --block_size 1024 \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 4e-4 \
    --node_index=0 \
    --gpu_per_node $PER_NODE_GPU \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_grad_norm 1.0 \
    --max_steps 1000000 \
    --warmup_steps 10000 \
    --save_steps 5000 \
    --seed 123
```

## 3. Inference

Please download the [datasets](https://zenodo.org/record/8062703) first. Unzip it and move it to `./data`.

```bash
# prepare model checkpoint and datasets
cd inference
bash run.sh
```

A demo bash script (run.sh) is shown:
```bash
# Change the arguments as required:
#   prefix: dataset type (codenet/tutorial/singleline)
#   output_dir: the output directory to save inference results
#   data_cache_dir: the output directory to save the data cache 
#   eval_data_path: the path of the test file
#   model_name_or_path: the path of the model to be evaluated

CUDA_VISIBLE_DEBVISES=0 python run.py \
    --prefix codenet \
    --output_dir ../../saved_models/inference \
    --data_cache_dir ../../saved_models/inference \
    --eval_data_path ../data/codenetmut_test.json \
    --model_name_or_path microsoft/codeexecutor \
    --block_size 1024 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --node_index 0 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_grad_norm 1.0 \
    --max_steps 1000 \
    --warmup_steps 10000 \
    --save_steps 5000 \
    --seed 123456
```

## 4. Downstream tasks

We apply CodeExecutor on code intelligence tasks, such as the Zero-shot Code-to-code Search task.
Here, we provide example code in which the baseline model is UniXcoder.

First, generate traces for the code-to-code search test set. We provide the prediction file `code_to_code_search_preds.txt` on [Zenodo](https://zenodo.org/record/8062703).

Or use the following script to generate the prediciton file (will be `../saved_models/code_to_code_search/preds.txt`).

```bash
# prepare model checkpoint and datasets
cd inference

CUDA_VISIBLE_DEBVISES=0 python run.py \
    --prefix codenet \
    --output_dir ../saved_models/code_to_code_search \
    --data_cache_dir ../saved_models/code_to_code_search \
    --eval_data_path ../data/code_to_code_search_test.json \
    --model_name_or_path microsoft/codeexecutor \
    --block_size 1024 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --node_index 0 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_grad_norm 1.0 \
    --max_steps 1000 \
    --warmup_steps 10000 \
    --save_steps 5000 \
    --seed 123456
```

Second, utilize the program outputs extracted from the execution trace generated by CodeExecutor to facilitate the code-to-code search task.

```bash
cd downstream
bash run.sh
```

A demo bash script (run.sh) is shown:
```bash
# Change the arguments as required:
#   trace_file: the path to the prediction file either downloaded or generated in the last step

source_lang=python
target_lang=python
python run.py \
    --model_name_or_path microsoft/unixcoder-base  \
    --query_data_file ../data/code_to_code_search_test.json \
    --candidate_data_file ../data/code_to_code_search_test.json \
    --trace_file ../data/code_to_code_search_preds.txt \
    --query_lang ${source_lang} \
    --candidate_lang ${target_lang} \
    --code_length 512 \
    --eval_batch_size 256 
```


# Reference

If you use this code or CodeExecutor, please consider citing us.

```
@article{liu2023code,
  title={Code Execution with Pre-trained Language Models},
  author={Liu, Chenxiao and Lu, Shuai and Chen, Weizhu and Jiang, Daxin and Svyatkovskiy, Alexey and Fu, Shengyu and Sundaresan, Neel and Duan, Nan},
  journal={arXiv preprint arXiv:2305.05383},
  year={2023}
}
```

