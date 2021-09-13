# Clone Detection

## Task Definition

Given two codes as the input, the task is to do binary classification (0/1), where 1 stands for semantic equivalence and 0 for others. Models are evaluated by F1 score.

## Updates

2021-9-13: We have update the evaluater script. Since it's a binary classification, we use binary F1 score instead of "macro" F1 score.

## Dataset

The dataset we use is [BigCloneBench](https://www.cs.usask.ca/faculty/croy/papers/2014/SvajlenkoICSME2014BigERA.pdf) and filtered following the paper [Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree](https://arxiv.org/pdf/2002.08653.pdf).

### Data Format

1. dataset/data.jsonl is stored in jsonlines format. Each line in the uncompressed file represents one function.  One row is illustrated below.

   - **func:** the function

   - **idx:** index of the example

2. train.txt/valid.txt/test.txt provide examples, stored in the following format:    idx1	idx2	label

### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Examples |
| ----- | :-------: |
| Train |  901,028  |
| Dev   |  415,416  |
| Test  |  415,416  |

You can get data using the following command.

```
unzip dataset.zip
```

## Evaluator

We provide a script to evaluate predictions for this task, and report F1 score

### Example

```bash
python evaluator/evaluator.py -a evaluator/answers.txt -p evaluator/predictions.txt
```

{'Recall': 0.25, 'Prediction': 0.5, 'F1': 0.3333333333333333}

### Input predictions

A predications file that has predictions in TXT format, such as evaluator/predictions.txt. For example:

```b
13653451	21955002	0
1188160	8831513	1
1141235	14322332	0
16765164	17526811	1
```

## Pipeline-GraphCodeBERT

We also provide a pipeline that fine-tunes GraphCodeBERT on this task. 
### Dependency

- pip install torch
- pip install transformers
- pip install tree_sitter
- pip sklearn

### Tree-sitter (optional)

If the built file "parser/my-languages.so" doesn't work for you, please rebuild as the following command:

```shell
cd parser
bash build.sh
cd ..
```

### Fine-tune

We use 4*V100-16G to fine-tune and 10% valid data to evaluate.


```shell
mkdir saved_models
python run.py \
    --output_dir=saved_models \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=dataset/train.txt \
    --eval_data_file=dataset/valid.txt \
    --test_data_file=dataset/test.txt \
    --epoch 1 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee saved_models/train.log
```

### Inference

We use full test data for inference. 

```shell
python run.py \
    --output_dir=saved_models \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_eval \
    --do_test \
    --train_data_file=dataset/train.txt \
    --eval_data_file=dataset/valid.txt \
    --test_data_file=dataset/test.txt \
    --epoch 1 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee saved_models/test.log
```

### Evaluation

```shell
python evaluator/evaluator.py -a dataset/test.txt -p saved_models/predictions.txt 2>&1| tee saved_models/score.log
```

## Result

The results on the test set are shown as below:

| Method        | Precision |  Recall   |    F1     |
| ------------- | :-------: | :-------: | :-------: |
| Deckard       |   0.93    |   0.02    |   0.03    |
| RtvNN         |   0.95    |   0.01    |   0.01    |
| CDLH          |   0.92    |   0.74    |   0.82    |
| ASTNN         |   0.92    |   0.94    |   0.93    |
| FA-AST-GMN    |   **0.96**    |   0.94    |   0.95    |
| CodeBERT      |   0.947   |   0.934   |   0.941   |
| GraphCodeBERT |  0.948 | **0.952** | **0.950** |

