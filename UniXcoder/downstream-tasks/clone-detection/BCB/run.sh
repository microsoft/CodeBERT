model=../../../../pretrained-model/UniXcoder-base
mkdir saved_models
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --model_name_or_path=$model \
    --do_train \
    --train_data_file=../../dataset/train.txt \
    --eval_data_file=../../dataset/valid.txt \
    --test_data_file=../../dataset/test.txt \
    --epoch 1 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee saved_models/train.log
    
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --model_name_or_path=$model \
    --do_eval \
    --do_test \
    --train_data_file=../../dataset/train.txt \
    --eval_data_file=../../dataset/valid.txt \
    --test_data_file=../../dataset/test.txt \
    --epoch 1 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee saved_models/test.log
    
python ../evaluator/evaluator.py -a ../../dataset/test.txt -p saved_models/predictions.txt 2>&1| tee saved_models/score.log
