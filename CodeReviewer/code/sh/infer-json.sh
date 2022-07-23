# batch size 6 for 16 GB GPU

mnt_dir="/home/v-zhuoli1/lzzz"

MASTER_HOST=localhost && echo MASTER_HOST: ${MASTER_HOST}
MASTER_PORT=23333 && echo MASTER_PORT: ${MASTER_PORT}
RANK=0 && echo RANK: ${RANK}
PER_NODE_GPU=1 && echo PER_NODE_GPU: ${PER_NODE_GPU}
WORLD_SIZE=1 && echo WORLD_SIZE: ${WORLD_SIZE}
NODES=1 && echo NODES: ${NODES}
NCCL_DEBUG=INFO

bash test_nltk.sh


# change break_cnt to truncate the number of examples (useful at debug time maybe)
#   --break_cnt -1 \  will keep the whole dataset 
python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_infer_msg.py  \
  --model_type codet5 \
  --add_lang_ids \
  --config_name ${mnt_dir}/PreViewer/saved_models_gen_shuai_link/checkpoints-21600-3.24 \
  --tokenizer_path ${mnt_dir}/PreViewer/pretrained_models/codet5 \
  --model_name_or_path ${mnt_dir}/PreViewer/saved_models_gen_shuai_link/checkpoints-21600-3.24 \
  --load_model_path ${mnt_dir}/PreViewer/saved_models_gen_shuai_link/checkpoints-21600-3.24 \
  --output_dir ${mnt_dir}/PreViewer/empty \
  --eval_file ${mnt_dir}/datasets/linkedin/test.jsonl \
  --out_file ${mnt_dir}/datasets/linkedin/test_out.jsonl \
  --max_source_length 512 \
  --max_target_length 128 \
  --eval_batch_size 12 \
  --beam_size 10 \
  --gpu_per_node=${PER_NODE_GPU} \
  --node_index=${RANK} \
  --seed 2233 \
  --raw_input \
  --break_cnt 20
