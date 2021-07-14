# Code Translation

## Task Definition

Code translation aims to migrate legacy software from one programming language in a platform toanother.
Given a piece of Java (C#) code, the task is to translate the code into C# (Java) version. 
Models are evaluated by BLEU scores and accuracy (exactly match).

## Dataset

The dataset is collected from several public repos, including Lucene(http://lucene.apache.org/), POI(http://poi.apache.org/), JGit(https://github.com/eclipse/jgit/) and Antlr(https://github.com/antlr/).

We collect both the Java and C# versions of the codes and find the parallel functions. After removing duplicates and functions with the empty body, we split the whole dataset into training, validation and test sets.

### Data Format

The dataset is in the "data" folder. Each line of the files is a function, and the suffix of the file indicates the programming language. You can get data using the following command:

```
unzip data.zip
```

### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Examples |
| ----- | :-------: |
| Train |  10,300   |
| Valid |    500    |
| Test  |   1,000   |

## Pipeline-GraphCodeBERT

### Dependency

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

### Fine-tune
We use 4*V100-16G to fine-tune. Taking Java to C# translation as example:

```shell
source=java
target=cs
lr=1e-4
batch_size=32
beam_size=10
source_length=320
target_length=256
output_dir=saved_models/$source-$target/
train_file=data/train.java-cs.txt.$source,data/train.java-cs.txt.$target
dev_file=data/valid.java-cs.txt.$source,data/valid.java-cs.txt.$target
epochs=100
pretrained_model=microsoft/graphcodebert-base

mkdir -p $output_dir
python run.py \
--do_train \
--do_eval \
--model_type roberta \
--source_lang $source \
--model_name_or_path $pretrained_model \
--tokenizer_name microsoft/graphcodebert-base \
--config_name microsoft/graphcodebert-base \
--train_filename $train_file \
--dev_filename $dev_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--num_train_epochs $epochs 2>&1| tee $output_dir/train.log
```

### Inference

We use full test data for inference. 

```shell
batch_size=64
dev_file=data/valid.java-cs.txt.$source,data/valid.java-cs.txt.$target
test_file=data/test.java-cs.txt.$source,data/test.java-cs.txt.$target
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

python run.py \
--do_test \
--model_type roberta \
--source_lang $source \
--model_name_or_path $pretrained_model \
--tokenizer_name microsoft/graphcodebert-base \
--config_name microsoft/graphcodebert-base \
--load_model_path $load_model_path \
--dev_filename $dev_file \
--test_filename $test_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--eval_batch_size $batch_size 2>&1| tee $output_dir/test.log
```



## Result

The results on the test set are shown as below:

Java to C#:

| Method         |   BLEU    | Acc (100%) |
| -------------- | :-------: | :--------: |
| Naive copy     |   18.54   |    0.0     |
| PBSMT          |   43.53   |    12.5    |
| Transformer    |   55.84   |    33.0    |
| Roborta (code) |   77.46   |    56.1    |
| CodeBERT       |   79.92   |    59.0    |
| GraphCodeBERT  | **80.58** |  **59.4**  |

C# to Java:

| Method         |   BLEU    | Acc (100%) |
| -------------- | :-------: | :--------: |
| Naive copy     |   18.69   |    0.0     |
| PBSMT          |   40.06   |    16.1    |
| Transformer    |   50.47   |    37.9    |
| Roborta (code) |   71.99   |    57.9    |
| CodeBERT       |   72.14   |    58.0    |
| GraphCodeBERT  | **72.64** |  **58.8**  |
