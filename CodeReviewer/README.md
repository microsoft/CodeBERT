# CodeReviewer

This repo provides the code for reproducing the experiments in [CodeReviewer: Pre-Training for Automating Code Review Activities](https://arxiv.org/abs/2203.09095). **CodeReviewer** is a model pre-trained with code change and code review data to support code review tasks.

The pre-trained checkpoint of CodeReivewer is available in [Huggingface](https://huggingface.co/microsoft/codereviewer). 

Our dataset is available in [Zenodo](https://zenodo.org/record/6900648).

## 1. Dependency

- conda install nltk
- conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
- conda install transformers


## 2. Brief Introduction

CodeReviewer supports for three related tasks: **Quality Estimation** (`cls` for short), **Comment Generation** (`msg` for short) and **Code Refinement** (`ref` for short).

Demo data:

``` python
{
    "old_file": "import torch",  # f1
    "diff_hunk": "@@ -1 +1,2 @@\n import torch\n +import torch.nn as nn",  # f1->f2
    "comment": "I don't think we need to import torch.nn here.",  # requirements for f2->f3
    "target": "import torch"  # f3
}
```

* Quality Estimation: input with "old_file" and "diff_hunk", we need to predict that whether the code change is not good and needs a comment.

* Comment Generation: input with "old_file" and "diff_hunk", we need to generate a comment for the change. An expected comment is as the "comment" above.

* Code Refinement: input with "old_file", "diff_hunk", and "comment", we need to change the code again according to the review comment. For the above example, as the comment indicated we don't need *import torch.nn*, we just delete this line of code here.

The model inputs are code change (old file and diff hunk) and review comment (optional according to task). Input data is preprocessed in `utils.py: ReviewExample` and wrapped to {`utils.py: CommentClsDataset, SimpleGenDataset, RefineDataset`}

## 3. Finetune/Inference

Before you start to run experiments with CodeReviewer, please download the [datasets](https://zenodo.org/record/6900648) first.

```bash
# prepare model checkpoint and datasets
cd code/sh
# adjust the arguments in the *sh* scripts
bash finetune-cls.sh
```

A demo bash script (finetune-cls.sh) is shown:
```bash
mnt_dir="/home/codereview"

# You may change the following block for multiple gpu training
MASTER_HOST=localhost && echo MASTER_HOST: ${MASTER_HOST}
MASTER_PORT=23333 && echo MASTER_PORT: ${MASTER_PORT}
RANK=0 && echo RANK: ${RANK}
PER_NODE_GPU=1 && echo PER_NODE_GPU: ${PER_NODE_GPU}
WORLD_SIZE=1 && echo WORLD_SIZE: ${WORLD_SIZE}
NODES=1 && echo NODES: ${NODES}
NCCL_DEBUG=INFO

bash test_nltk.sh


# Change the arguments as required:
#   model_name_or_path, load_model_path: the path of the model to be finetuned
#   eval_file: the path of the evaluation data
#   output_dir: the directory to save finetuned model (not used at infer/test time)
#   out_file: the path of the output file
#   train_file_name: can be a directory contraining files named with "train*.jsonl"

python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_finetune_cls.py  \
  --train_epochs 30 \
  --model_name_or_path microsoft/codereviewer \
  --output_dir ../../save/cls \
  --train_filename ../../dataset/Diff_Quality_Estimation \
  --dev_filename ../../dataset/Diff_Quality_Estimation/cls-valid.jsonl \
  --max_source_length 512 \
  --max_target_length 128 \
  --train_batch_size 12 \
  --learning_rate 3e-4 \
  --gradient_accumulation_steps 3 \
  --mask_rate 0.15 \
  --save_steps 3600 \
  --log_steps 100 \
  --train_steps 120000 \
  --gpu_per_node=${PER_NODE_GPU} \
  --node_index=${RANK} \
  --seed 2233 
```


## 4. File structure
```
.
├── bleu.py                 # demo code for BLEU evaluation
├── configs.py
├── evaluator               # copied from CodeXGlue for BLEU evaluation
├── models.py               # CodeReviewer model
├── run_finetune_xxx.py     # finetune script - xxx in {cls, msg, gen}
├── run_infer_msg.py        # inference script for comment generation task
├── run_test_xxx.py         # test script - xxx in {cls, msg, gen}
├── sh/xx.sh                # bash script for running finetune and test scripts with arguments
│   ├── finetune-xxx.sh
│   ├── infer-json.sh
│   ├── test-xxx.sh
│   ├── test_nltk.sh
└── utils.py                # utils for data preprocessing
```

# Reference
If you use this code or CodeReviewer, please consider citing us.

<pre><code>@article{li2022codereviewer,
  title={CodeReviewer: Pre-Training for Automating Code Review Activities},
  author={Li, Zhiyu and Lu, Shuai and Guo, Daya and Duan, Nan and Jannu, Shailesh and Jenks, Grant and Majumder, Deep and Green, Jared and Svyatkovskiy, Alexey and Fu, Shengyu and others},
  journal={arXiv preprint arXiv:2203.09095},
  year={2022}
}</code></pre>



