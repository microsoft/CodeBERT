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

from model import Model
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, RobertaTokenizer

from dataset import TextDataset

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)        
        

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
        
def train(args, train_datasets, eval_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_samplers = [RandomSampler(train_dataset) for train_dataset in train_datasets]
    
    train_dataloaders = [cycle(DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,drop_last = True,num_workers = 0)) for train_dataset,train_sampler in zip(train_datasets,train_samplers)]
      
    model.to(args.device)
    
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()     
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr = args.learning_rate, eps = args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.warmup_steps,
                                                num_training_steps = args.max_steps)
                                                
    # use reintialized scheduler and opitmizer actually
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last, map_location="cpu"))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last, map_location="cpu"))  
  
        
    if args.local_rank == 0:
        torch.distributed.barrier()    
    
    # using fp16 to accelerate training
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [args.local_rank%args.gpu_per_node],
                                                          output_device = args.local_rank%args.gpu_per_node,
                                                          find_unused_parameters = True)

    # Train!
    logger.warning("***** Running training *****")
    logger.warning("  Num examples = %d",sum([len(train_dataset) for train_dataset in train_datasets]) * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.warning("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.warning("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1)) 
    logger.warning("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.warning("  Total optimization steps = %d", args.max_steps)


    global_step = args.start_step
    losses, contras_losses, align_losses, dual_losses, step = [], [], [], [], 0

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    probs = [0.34,0.33,0.33]
    while True:
        train_dataloader = np.random.choice(train_dataloaders, 1, p=probs)[0]
        batch = next(train_dataloader)

        model.train()
        step+=1
        
        # forward
        dual_gen_ids, dual_gen_type_ids =[x.to(args.device) for x in batch]          
        loss, dual_loss, align_loss, contras_loss = model(dual_gen_ids, dual_gen_type_ids)
        
        # store loss
        losses.append(loss.item())
        if contras_loss != 0:
            contras_losses.append(contras_loss)   
        if align_loss != 0:
            align_losses.append(align_loss)
        if dual_loss != 0:
            dual_losses.append(dual_loss)                  
        if args.n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu parallel training
        
        # backward
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
            
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        # update model
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  
            global_step += 1
            
            if global_step %100 == 0:
                logger.warning("steps: %s dual: %s", global_step, 
                        round(np.mean(dual_losses),3),
                    )
                losses,  contras_losses, align_losses, dual_losses = [], [], [], []
            
            # evaluate model and save model
            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:  
                
                checkpoint_prefix = 'checkpoint'
                results = evaluate(args, model, tokenizer,eval_dataset)
                for key, value in results.items():
                    logger.warning("  %s = %s", key, round(value,6))      
                    
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, '{}-{}-{}'.format(checkpoint_prefix, 
                                                                            global_step, 
                                                                            round(results['loss'], 6)))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module.encoder if hasattr(model,'module') else model.encoder
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.warning("Saving model checkpoint to %s", output_dir)


                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save.save_pretrained(last_output_dir)
                tokenizer.save_pretrained(last_output_dir)
                idx_file = os.path.join(last_output_dir, 'idx_file.txt')
                with open(idx_file, 'w', encoding='utf-8') as idxf:
                    idxf.write(str(0) + '\n')

                torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
                logger.warning("Saving optimizer and scheduler states to %s", last_output_dir)

                step_file = os.path.join(last_output_dir, 'step_file.txt')
                with open(step_file, 'w', encoding='utf-8') as stepf:
                    stepf.write(str(global_step) + '\n')


        if args.max_steps > 0 and global_step > args.max_steps:
            break


def evaluate(args, model, tokenizer, eval_dataset,prefix=""):
    """ Evaluate the model """
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_dataloader = DataLoader(eval_dataset, 
                                 sampler = SequentialSampler(eval_dataset), 
                                 batch_size = args.eval_batch_size,
                                 num_workers = 4, 
                                 drop_last = True)

    # Eval!
    logger.warning("***** Running evaluation *****")
    logger.warning("  Num examples = %d", len(eval_dataset))
    logger.warning("  Batch size = %d", args.eval_batch_size)


    model.eval()
    losses, contras_losses, align_losses, dual_losses = [], [], [], []
    for batch in eval_dataloader:
        dual_gen_ids, dual_gen_type_ids =[x.to(args.device) for x in batch]          
        with torch.no_grad():      
            loss, dual_loss, align_loss, contras_loss = model(dual_gen_ids, dual_gen_type_ids)
            losses.append(loss.item())
            if contras_loss != 0:
                 contras_losses.append(contras_loss)   
            if align_loss != 0:
                align_losses.append(align_loss)
            if dual_loss != 0:
                 dual_losses.append(dual_loss)                         

    result = {
        "loss": round(np.mean(losses),4),
        "dual": round(np.mean(dual_losses),4),
    }

    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_path", default=None, type=str, required=True,
                        help="The input training data path")
    parser.add_argument("--another_train_data_path", default=None, type=str, required=True,
                        help="The input training data path")
    parser.add_argument("--third_train_data_path", default=None, type=str, required=True,
                        help="The input training data path")
    parser.add_argument("--eval_data_path", default=None, type=str,
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

    args.start_step = 0
    model_name_or_path = args.model_name_or_path
       
    tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path if args.tokenizer_name is None else args.tokenizer_name)    
    config = RobertaConfig.from_pretrained(model_name_or_path if args.config_name is None else args.config_name)
    model = RobertaModel.from_pretrained(model_name_or_path,config=config)  
    # add special tokens
    special_tokens_list = ['<line>','<state>','</state>','<dictsep>','<output>','<function>','<singleline>','<tutorial>','<codenet>','<indent>','<dedent>']
    for i in range(200):
        special_tokens_list.append('<' + str(i) + '>')
    special_tokens_dict = {'additional_special_tokens': special_tokens_list}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))   

    model = Model(model,config,tokenizer,args)
    
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
    train_datasets = []
    train_datasets.append(TextDataset(tokenizer, args, args.train_data_path, local_rank, world_size, logger, "train", prefix="codenet"))
    train_datasets.append(TextDataset(tokenizer, args, args.another_train_data_path, local_rank, world_size, logger, "train", prefix="tutorial"))
    train_datasets.append(TextDataset(tokenizer, args, args.third_train_data_path, local_rank, world_size, logger, "train", prefix="singleline"))
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_path, local_rank, world_size, logger, "eval","codenet")

    # Training
    train(args, train_datasets, eval_dataset, model, tokenizer)



if __name__ == "__main__":
    main()





