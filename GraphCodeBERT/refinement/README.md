# CodeXGLUE -- Code-To-Text

## Task Definition

The task is to generate natural language comments for a code, and evaluted by [smoothed bleu-4](https://www.aclweb.org/anthology/C04-1072.pdf) score.

## Dataset

The dataset we use comes from [CodeSearchNet](https://arxiv.org/pdf/1909.09436.pdf) and we filter the dataset as the following:

- Remove examples that codes cannot be parsed into an abstract syntax tree.
- Remove examples that #tokens of documents is < 3 or >256
- Remove examples that documents contain special tokens (e.g. <img ...> or https:...)
- Remove examples that documents are not English.

### Download data and preprocess

```shell
unzip dataset.zip
cd dataset
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/ruby.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/javascript.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/go.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/php.zip

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

### Data Format

After preprocessing dataset, you can obtain three .jsonl files, i.e. train.jsonl, valid.jsonl, test.jsonl

For each file, each line in the uncompressed file represents one function.  One row is illustrated below.

  - **repo:** the owner/repo

  - **path:** the full path to the original file

  - **func_name:** the function or method name

  - **original_string:** the raw string before tokenization or parsing

  - **language:** the programming language

  - **code/function:** the part of the `original_string` that is code

  - **code_tokens/function_tokens:** tokenized version of `code`

  - **docstring:** the top-level comment or docstring, if it exists in the original string

  - **docstring_tokens:** tokenized version of `docstring`

### Data Statistic

| Programming Language | Training |  Dev   |  Test  |
| :------------------- | :------: | :----: | :----: |
| Python               | 251,820  | 13,914 | 14,918 |
| PHP                  | 241,241  | 12,982 | 14,014 |
| Go                   | 167,288  | 7,325  | 8,122  |
| Java                 | 164,923  | 5,183  | 10,955 |
| JavaScript           |  58,025  | 3,885  | 3,291  |
| Ruby                 |  24,927  | 1,400  | 1,261  |

## Evaluator

We provide a script to evaluate predictions for this task, and report smoothed bleu-4 score.

### Example

```shell
python evaluator/evaluator.py evaluator/reference.txt < evaluator/predictions.txt
```

Total: 5
9.554726113590661

## Pipeline-CodeBERT

We also provide a pipeline that fine-tunes [CodeBERT](https://arxiv.org/pdf/2002.08155.pdf) on this task. The encoder is CodeBERT and the decoder is 6-layers Transformer.

### Dependency

- python 3.6 or 3.7
- torch==1.4.0
- transformers>=2.5.0

### Fine-tune

To fine-tune encoder-decoder on the dataset

```shell
cd code
lang=ruby #programming language
lr=5e-5
batch_size=32
beam_size=10
source_length=256
target_length=128
data_dir=../dataset
output_dir=model/$lang
train_file=$data_dir/$lang/train.jsonl
dev_file=$data_dir/$lang/valid.jsonl
epochs=10 
pretrained_model=microsoft/codebert-base #Roberta: roberta-base

python run.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs
```


### Inference

```shell
batch_size=64
dev_file=$data_dir/$lang/valid.jsonl
test_file=$data_dir/$lang/test.jsonl
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

python run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size
```

### Evaluation

```shell
python ../evaluator/evaluator.py model/$lang/test_1.gold < model/$lang/test_1.output
```

## Result

The results on the test set are shown as below:

| Model       |   Ruby    | Javascript |    Go     |  Python   |   Java    |    PHP    |  Overall  |
| ----------- | :-------: | :--------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| Seq2Seq     |   9.64    |   10.21    |   13.98   |   15.93   |   15.09   |   21.08   |   14.32   |
| Transformer |   11.18   |   11.59    |   16.38   |   15.81   |   16.26   |   22.12   |   15.56   |
| [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf)     |   11.17   |   11.90    |   17.72   |   18.14   |   16.47   |   24.02   |   16.57   |
| [CodeBERT](https://arxiv.org/pdf/2002.08155.pdf)    | **12.16** | **14.90**  | **18.07** | **19.06** | **17.65** | **25.16** | **17.83** |


## Reference
<pre><code>@article{husain2019codesearchnet,
  title={Codesearchnet challenge: Evaluating the state of semantic code search},
  author={Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
  journal={arXiv preprint arXiv:1909.09436},
  year={2019}
}</code></pre>
