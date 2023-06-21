# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import pickle
import random
import torch
import numpy as np
from itertools import cycle
import json
from collections import Counter

from model import Seq2Seq
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, RobertaTokenizer

from dataset import TextDataset
from metric import compute_metrics, compute_singleline_metrics


logger = logging.getLogger(__name__)        
        

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
def eval(args, model, tokenizer, eval_dataset,prefix=""):
    model.to(args.device)

    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_dataloader = DataLoader(eval_dataset, 
                                 sampler = SequentialSampler(eval_dataset), 
                                 batch_size = args.eval_batch_size,
                                 num_workers = 4, 
                                 drop_last = False)

    # Eval!
    logger.warning("***** Running evaluation *****")
    logger.warning("  Num examples = %d", len(eval_dataset))
    logger.warning("  Batch size = %d", args.eval_batch_size)          

    model.eval() 
    pred_list = []
    gold_list = []
    for batch in eval_dataloader:
        source_ids, target_ids,gold_ids =[x.to(args.device) for x in batch]                
        with torch.no_grad():
            preds = model(source_ids)   
            # convert ids to text
            for i,pred in enumerate(preds):
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = tokenizer.decode(t,clean_up_tokenization_spaces=False)

                gold = gold_ids[i].cpu().numpy()
                gold = list(gold) 
                if 1 in gold:
                    gold = gold[:gold.index(1)]
                gold = gold[1:-1]# <mask0>    </s>
                gold = tokenizer.decode(gold,clean_up_tokenization_spaces=False)

                pred_list.append(text)
                gold_list.append(gold)

    with open(args.output_dir+"/preds.txt",'w') as f:
        for i in pred_list:
            f.write(str(i) + '\n')  
    with open(args.output_dir+"/golds.txt",'w') as f:
        for i in gold_list:
            f.write(str(i) + '\n')  

    if args.prefix == "singleline":
        metric_list = compute_singleline_metrics(pred_list, gold_list)
        logger.warning(f"Trace Accuracy: {metric_list[0]}")
        logger.warning(f"Identifier Precision: {metric_list[1]}")
        logger.warning(f"Identifier Recall: {metric_list[2]}")
        logger.warning(f"Identifier F1: {metric_list[3]}")
    else:
        metric_list = compute_metrics(pred_list, gold_list)
        logger.warning(f"Output Accuracy: {metric_list[0]}")
        logger.warning(f"Trace Accuracy: {metric_list[1]}")
        logger.warning(f"Line Precision: {metric_list[2]}")
        logger.warning(f"Line Recall: {metric_list[3]}")
        logger.warning(f"Line F1: {metric_list[4]}")
        logger.warning(f"Identifier Precision: {metric_list[5]}")
        logger.warning(f"Identifier Recall: {metric_list[6]}")
        logger.warning(f"Identifier F1: {metric_list[7]}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="", type=str,
                        help="The input data prefix.")

    ## Required parameters
    parser.add_argument("--train_data_path", default=None, type=str,
                        help="The input training data path")
    parser.add_argument("--eval_data_path", default=None, type=str,required=True,
                        help="The input evaluating data path")    
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--data_cache_dir", default=None, type=str, required=True,
                        help="The output directory where data cache will be written.")
    parser.add_argument("--reload_dir", default=None, type=str, 
                        help="The directory where the model checkpoints will be reloaded from.")

    ## Other parameters
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default=None, type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--block_size", default=1024, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                        "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")  
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--node_index", type=int, default=-1,
                        help="For distributed training: local_rank")    
    parser.add_argument("--gpu_per_node", type=int, default=-1,
                        help="For distributed training: local_rank")     
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    parser.add_argument("--max_source_length", default=256, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=768, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search") 

    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        import datetime
        torch.distributed.init_process_group(backend='nccl',timeout=datetime.timedelta(0,1800000))
        args.local_rank+=args.node_index*args.gpu_per_node
        args.n_gpu = 1

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.INFO) #logging.WARN
    
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    args.log_file = os.path.join(args.output_dir, 'log.txt')
    if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    if os.path.exists(args.log_file):
            logfile = logging.FileHandler(args.log_file, 'a')
    else:
        logfile = logging.FileHandler(args.log_file, 'w')
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%m/%d/%Y %H:%M:%S %p')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    if args.local_rank == 0:
        torch.distributed.barrier() 
    
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)


    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier() 

    model_name_or_path = args.model_name_or_path
    tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path if args.tokenizer_name is None else args.tokenizer_name)    
    config = RobertaConfig.from_pretrained(model_name_or_path if args.config_name is None else args.config_name)
    config.is_decoder = True
    encoder = RobertaModel.from_pretrained(model_name_or_path,config=config)  
    model = Seq2Seq(encoder=encoder,decoder=encoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],eos_id=tokenizer.sep_token_id)
    
    
    
    if args.local_rank == 0:
        torch.distributed.barrier()  

    logger.warning("Training/evaluation parameters %s", args)
    
    if args.local_rank == -1:
        local_rank = 0
        world_size = 1
    else:
        local_rank = args.local_rank
        world_size = torch.distributed.get_world_size()
    
    # reload and preprocess data
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_path, local_rank, world_size, logger, "eval",args.prefix)

    # eval
    eval(args, model, tokenizer,eval_dataset)


if __name__ == "__main__":
    main()





