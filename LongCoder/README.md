# LongCoder

This repo will provide the code for reproducing the experiments on LCC datasets in [LongCoder: A Long-Range Pre-trained Language Model for Code Completion](https://arxiv.org/abs/2306.14893). LongCoder is a sparse and efficient pre-trained Transformer model for long code modeling.

## 1. Dependency

- pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
- pip install --upgrade  transformers fuzzywuzzy tree_sitter datasets

## 2. Dataset
In this repo, the LCC dataset will be automatically downloaded when running the fine-tuning script. If you want to download LCC datasets by yourself, you can find them in the following links:
```
https://huggingface.co/datasets/microsoft/LCC_python
https://huggingface.co/datasets/microsoft/LCC_java
https://huggingface.co/datasets/microsoft/LCC_csharp
```
## 3. Fine-Tune Setting
Here we provide fine-tune settings for code completion on LCC datasets in C# programming language, whose results are reported in the paper.

Note that it requires 8 v100-32G GPUs, and you can adjust batch size or source length based on your requirements.

```shell
lang=csharp #csharp, python, java
lr=2e-4
batch_size=16
beam_size=5
source_length=3968
target_length=128
global_length=64
window_size=512
epochs=10
output_dir=saved_models/$lang
mkdir -p $output_dir

python run.py \
--do_train \
--do_eval \
--lang $lang \
--output_dir $output_dir \
--model_name_or_path microsoft/longcoder-base \
--filename microsoft/LCC_$lang \
--max_source_length $source_length \
--max_target_length $target_length \
--max_global_length $global_length \
--window_size $window_size \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--num_train_epochs $epochs  2>&1| tee $output_dir/train.log
```

## 4. Evaluating LongCoder

```shell
lang=csharp #csharp, python, java
batch_size=16
beam_size=5
source_length=3968
target_length=128
global_length=64
window_size=512
output_dir=saved_models/$lang
reload_model=$output_dir/checkpoint-best-acc/model.bin

python run.py \
--do_test \
--lang $lang \
--load_model_path $reload_model \
--output_dir $output_dir \
--model_name_or_path microsoft/longcoder-base \
--filename microsoft/LCC_$lang \
--max_source_length $source_length \
--max_target_length $target_length \
--max_global_length $global_length \
--window_size $window_size \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--num_train_epochs $epochs 2>&1| tee $output_dir/test.log
```

# Reference
If you use this code or LongCoder, please consider citing us.

<pre><code>@article{longcoder,
    title={LongCoder: A Long-Range Pre-trained Language Model for Code Completion},
    author={Daya Guo and Canwen Xu and Nan Duan and Jian Yin and Julian McAuley},
    journal={arXiv preprint arXiv:2306.14893},
    year={2023}
}</code></pre>




