lr=1e-4
batch_size=32
beam_size=10
source_length=320
target_length=256
output_dir=translation/model/$1-$2/
train_file=translation/data/train.java-cs.txt.$1,translation/data/train.java-cs.txt.$2
dev_file=translation/data/valid.java-cs.txt.$1,translation/data/valid.java-cs.txt.$2
epochs=100
pretrained_model=../pre-train/saved_models/graphcodebert_balance/checkpoint-200000-0.570868

CUDA_VISIBLE_DEVICES=$3 python -m translation.run --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --tokenizer_name roberta-base --config_name roberta-base --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs 2>&1| tee $output_dir/train.log


batch_size=64
dev_file=translation/data/valid.java-cs.txt.$1,translation/data/valid.java-cs.txt.$2
test_file=translation/data/test.java-cs.txt.$1,translation/data/test.java-cs.txt.$2
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

CUDA_VISIBLE_DEVICES=$3 python -m translation.run --do_test --model_type roberta --model_name_or_path $pretrained_model --tokenizer_name roberta-base --config_name roberta-base --load_model_path $test_model --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size 2>&1| tee $output_dir/test.log
