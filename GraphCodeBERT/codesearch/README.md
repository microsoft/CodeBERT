

# Code Search

## Data Preprocess

Different from the setting of [CodeSearchNet](husain2019codesearchnet), the answer of each query is retrieved from the whole development and testing code corpus instead of 1,000 candidate codes. Besides, we observe that some queries contain content unrelated to the code, such as a link ``http://..." that refers to external resources.  Therefore, we filter following examples to improve the quality of the dataset. 

- Remove comments in the code

- Remove examples that codes cannot be parsed into an abstract syntax tree.

- Remove examples that #tokens of documents is < 3 or >256

- Remove examples that documents contain special tokens (e.g. <img ...> or https:...)

- Remove examples that documents are not English.

Data statistic about the cleaned dataset for code document generation is shown in this Table.

| PL         | Training |  Dev   |  Test  | Candidates code |
| :--------- | :------: | :----: | :----: | :-------------: |
| Python     | 251,820  | 13,914 | 14,918 |     43,827      |
| PHP        | 241,241  | 12,982 | 14,014 |     52,660      |
| Go         | 167,288  | 7,325  | 8,122  |     28,120      |
| Java       | 164,923  | 5,183  | 10,955 |     40,347      |
| JavaScript |  58,025  | 3,885  | 3,291  |     13,981      |
| Ruby       |  24,927  | 1,400  | 1,261  |      4,360      |

You can download and preprocess data using the following command.
```shell
unzip dataset.zip
cd dataset
bash run.sh 
cd ..
```

## Dependency 

- pip install torch
- pip install transformers
- pip install tree_sitter

### Tree-sitter (optional)

If the built file "parser/my-languages.so" doesn't work for you, please rebuild as the following command:

```shell
cd parser
bash build.sh
cd ..
```

## Fine-Tune

We fine-tuned the model on 2*V100-16G GPUs. 
```shell
lang=ruby
mkdir -p ./saved_models/$lang
python run.py \
    --output_dir=./saved_models/$lang \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --lang=$lang \
    --do_train \
    --train_data_file=dataset/$lang/train.jsonl \
    --eval_data_file=dataset/$lang/valid.jsonl \
    --test_data_file=dataset/$lang/test.jsonl \
    --codebase_file=dataset/$lang/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --data_flow_length 64 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1| tee saved_models/$lang/train.log
```
## Inference and Evaluation

```shell
lang=ruby
python run.py \
    --output_dir=./saved_models/$lang \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --lang=$lang \
    --do_eval \
    --do_test \
    --train_data_file=dataset/$lang/train.jsonl \
    --eval_data_file=dataset/$lang/valid.jsonl \
    --test_data_file=dataset/$lang/test.jsonl \
    --codebase_file=dataset/$lang/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --data_flow_length 64 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1| tee saved_models/$lang/test.log
```

## Results	

The results on the filtered dataset are shown in this Table:

| Model          |   Ruby    | Javascript |    Go     |  Python   |   Java    |    PHP    |  Overall  |
| -------------- | :-------: | :--------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| NBow           |   0.162   |   0.157    |   0.330   |   0.161   |   0.171   |   0.152   |   0.189   |
| CNN            |   0.276   |   0.224    |   0.680   |   0.242   |   0.263   |   0.260   |   0.324   |
| BiRNN          |   0.213   |   0.193    |   0.688   |   0.290   |   0.304   |   0.338   |   0.338   |
| SelfAtt        |   0.275   |   0.287    |   0.723   |   0.398   |   0.404   |   0.426   |   0.419   |
| RoBERTa        |   0.587   |   0.517    |   0.850   |   0.587   |   0.599   |   0.560   |   0.617   |
| RoBERTa (code) |   0.628   |   0.562    |   0.859   |   0.610   |   0.620   |   0.579   |   0.643   |
| CodeBERT       |   0.679   |   0.620    |   0.882   |   0.672   |   0.676   |   0.628   |   0.693   |
| GraphCodeBERT  | **0.703** | **0.644**  | **0.897** | **0.692** | **0.691** | **0.649** | **0.713** |


## Model and Demo
A pretrained model, additional training script with dataset, and demo of a finetuned CodeBERT model for the task of Code Search can be found here: https://drive.google.com/file/d/1ZO-xVIzGcNE6Gz9DEg2z5mIbBv4Ft1cK/view.
