pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install --upgrade scipy transformers tqdm fuzzywuzzy tree_sitter datasets

lang=$1 #programming language
lr=2e-4
batch_size=16
beam_size=5
source_length=3968
target_length=128
global_length=64
window_size=512
output_dir=saved_models/$1
epochs=10
pretrained_model=microsoft/longcoder-base

mkdir -p $output_dir

python run.py \
--do_train \
--do_eval \
--lang $1 \
--output_dir $output_dir \
--model_name_or_path $pretrained_model \
--filename microsoft/LCC_$1 \
--max_source_length $source_length \
--max_target_length $target_length \
--max_global_length $global_length \
--window_size $window_size \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--num_train_epochs $epochs  2>&1| tee $output_dir/train.log





reload_model=$output_dir/checkpoint-best-acc/model.bin
python run.py \
--do_test \
--lang $1 \
--load_model_path $reload_model \
--model_name_or_path $pretrained_model \
--filename microsoft/LCC_$1 \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--max_global_length $global_length \
--window_size $window_size \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--num_train_epochs $epochs 2>&1| tee $output_dir/test.log
