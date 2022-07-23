# batch size 6 for 16 GB GPU

mnt_dir="/home/v-zhuoli1/lzzz"

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
#   config_name, model_name_or_path, load_model_path: the path of the model to be finetuned
#   tokenizer_path: the path of the tokenizer
#   eval_file: the path of the evaluation data
#   output_dir: the directory to save finetuned model (not used at infer/test time)
#   out_file: the path of the output file
#   train_file_name: can be a directory contraining files named with "train*.jsonl"
#   raw_input: to select the preprocess method, set to True in this task

python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_finetune_msg.py  \
  --model_type codet5 \
  --add_lang_ids \
  --train_epochs 30 \
  --config_name ${mnt_dir}/PreViewer/saved_models_codet5_shuai/save_codet5/checkpoints-245000-3.97 \
  --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
  --model_name_or_path ${mnt_dir}/PreViewer/saved_models_codet5_shuai/save_codet5/checkpoints-245000-3.97 \
  --load_model_path ${mnt_dir}/PreViewer/saved_models_codet5_shuai/save_codet5/checkpoints-245000-3.97 \
  --output_dir ${mnt_dir}/PreViewer/saved_models_gen_shuai_link \
  --train_filename ${mnt_dir}/LinkedIn \
  --dev_filename ${mnt_dir}/LinkedIn/valid.jsonl \
  --max_source_length 512 \
  --max_target_length 128 \
  --train_batch_size 6 \
  --learning_rate 3e-4 \
  --gradient_accumulation_steps 3 \
  --mask_rate 0.15 \
  --save_steps 1800 \
  --log_steps 100 \
  --train_steps 60000 \
  --gpu_per_node=${PER_NODE_GPU} \
  --node_index=${RANK} \
  --seed 2233 \
  --raw_input \

  # --model_type codet5 \
  # --add_lang_ids \
  # --train_epochs 30 \
  # --config_name ${mnt_dir}/PreViewer/saved_models_codet5_shuai/save_codet5/checkpoints-245000-3.97 \
  # --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
  # --model_name_or_path ${mnt_dir}/PreViewer/saved_models_codet5_shuai/save_codet5/checkpoints-245000-3.97 \
  # --load_model_path ${mnt_dir}/PreViewer/saved_models_codet5_shuai/save_codet5/checkpoints-245000-3.97 \
  # --output_dir ${mnt_dir}/PreViewer/saved_models_gen_shuai_2 \

  # --model_type t5 \
  # --add_lang_ids \
  # --train_epochs 30 \
  # --config_name ${mnt_dir}/Tufano/pytorch \
  # --tokenizer_path ${mnt_dir}/Tufano/pytorch/TokenizerModel.model \
  # --model_name_or_path ${mnt_dir}/Tufano/pytorch \
  # --output_dir ${mnt_dir}/PreViewer/saved_models_gen_tufano_2 \
  # --raw_input


  # --model_type codet5 \
  # --add_lang_ids \
  # --train_epochs 30 \
  # --config_name ${mnt_dir}/PreViewer/pretrained_models/codet5 \
  # --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
  # --model_name_or_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
  # --output_dir ${mnt_dir}/PreViewer/saved_models_gen_codet5_2 \
  # --raw_input