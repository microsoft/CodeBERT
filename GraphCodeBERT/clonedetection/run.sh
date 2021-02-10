pip install --user transformers tree_sitter tqdm sklearn > detection/log.txt 2>&1
mkdir detection/logs
python -m detection.run \
    --output_dir=detection/saved_models \
    --model_type=roberta \
    --config_name=roberta-base \
    --model_name_or_path=../pre-train/saved_models/graphcodebert_balance/checkpoint-200000-0.570868 \
    --tokenizer_name=roberta-base \
    --do_train \
    --train_data_file=detection/dataset/train.txt \
    --eval_data_file=detection/dataset/valid.txt \
    --test_data_file=detection/dataset/test.txt \
    --epoch 1 \
    --block_size 640 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee detection/logs/train.log
    
    
    
python -m detection.run \
    --output_dir=detection/saved_models \
    --model_type=roberta \
    --config_name=roberta-base \
    --model_name_or_path=../pre-train/saved_models/graphcodebert_balance/checkpoint-200000-0.570868 \
    --tokenizer_name=roberta-base \
    --do_eval \
    --do_test \
    --train_data_file=detection/dataset/train.txt \
    --eval_data_file=detection/dataset/valid.txt \
    --test_data_file=detection/dataset/test.txt \
    --epoch 1 \
    --block_size 640 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee detection/logs/test.log
    
    
python detection/evaluator/evaluator.py -a detection/dataset/test.txt -p detection/saved_models/predictions.txt 2>&1| tee detection/logs/score.log
