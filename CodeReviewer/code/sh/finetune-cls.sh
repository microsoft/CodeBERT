# batch size 12 for 16 GB GPU

mnt_dir="/home/codereview"

# You may change the following block for multiple gpu training
MASTER_HOST=localhost && echo MASTER_HOST: ${MASTER_HOST}
MASTER_PORT=23333 && echo MASTER_PORT: ${MASTER_PORT}
RANK=0 && echo RANK: ${RANK}
PER_NODE_GPU=1 && echo PER_NODE_GPU: ${PER_NODE_GPU}
WORLD_SIZE=1 && echo WORLD_SIZE: ${WORLD_SIZE}
NODES=1 && echo NODES: ${NODES}
NCCL_DEBUG=INFO

bash test_nltk.sh


# Change the arguments as required:
#   model_name_or_path, load_model_path: the path of the model to be finetuned
#   eval_file: the path of the evaluation data
#   output_dir: the directory to save finetuned model (not used at infer/test time)
#   out_file: the path of the output file
#   train_file_name: can be a directory contraining files named with "train*.jsonl"

python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_finetune_cls.py  \
  --train_epochs 30 \
  --model_name_or_path microsoft/codereviewer \
  --output_dir ../../save/cls \
  --train_filename ../../dataset/Diff_Quality_Estimation \
  --dev_filename ../../dataset/Diff_Quality_Estimation/cls-valid.jsonl \
  --max_source_length 512 \
  --max_target_length 128 \
  --train_batch_size 12 \
  --learning_rate 3e-4 \
  --gradient_accumulation_steps 3 \
  --mask_rate 0.15 \
  --save_steps 3600 \
  --log_steps 100 \
  --train_steps 120000 \
  --gpu_per_node=${PER_NODE_GPU} \
  --node_index=${RANK} \
  --seed 2233 
