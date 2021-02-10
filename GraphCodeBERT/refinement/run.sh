scale=$1
lr=1e-4
batch_size=32
beam_size=10
source_length=320
target_length=256
output_dir=refinement/model/$scale/
train_file=refinement/data/$scale/train.buggy-fixed.buggy,refinement/data/$scale/train.buggy-fixed.fixed
dev_file=refinement/data/$scale/valid.buggy-fixed.buggy,refinement/data/$scale/valid.buggy-fixed.fixed
epochs=50 
pretrained_model=../pre-train/saved_models/graphcodebert_balance/checkpoint-200000-0.570868

CUDA_VISIBLE_DEVICES=$2 python -m refinement.run --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --tokenizer_name roberta-base --config_name roberta-base --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs 2>&1| tee $output_dir/train-0.1.log


batch_size=64
dev_file=refinement/data/$scale/valid.buggy-fixed.buggy,refinement/data/$scale/valid.buggy-fixed.fixed
test_file=refinement/data/$scale/test.buggy-fixed.buggy,refinement/data/$scale/test.buggy-fixed.fixed
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

CUDA_VISIBLE_DEVICES=$2 python -m refinement.run --do_test --model_type roberta --model_name_or_path $pretrained_model --tokenizer_name roberta-base --config_name roberta-base --load_model_path $test_model --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size 2>&1| tee $output_dir/test-0.1.log
