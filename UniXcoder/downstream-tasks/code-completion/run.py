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

from __future__ import absolute_import
import os
import sys
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from fuzzywuzzy import fuzz
import re
import multiprocessing
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
              RobertaConfig, RobertaModel, RobertaTokenizer)  
cpu_cont = 16
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def read_examples(filename):
    """Read examples from filename."""
    examples=[]
    lang = filename.split("/")[-2]
    
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if ".txt" in filename:
                inputs=line.strip().replace("<EOL>","</s>").split()
                inputs=inputs[1:]
                inputs=" ".join(inputs)
                outputs=[]
            else:
                js=json.loads(line)
                inputs=js["input"].replace("<EOL>","</s>").split()
                inputs=inputs[1:]
                inputs=" ".join(inputs)                
                outputs=js["gt"]
                if 'id' in js:
                    idx = js['id']  
            examples.append(
                Example(
                        idx = idx,
                        source = inputs,
                        target = outputs,
                        ) 
            )
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
    ):
        self.example_id = example_id
        self.source_ids = source_ids  
        
def post_process(code):
    code = code.replace("<string","<STR_LIT").replace("<number","<NUM_LIT").replace("<char","<CHAR_LIT")
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    lits = re.findall(pattern, code)
    for lit in lits:
        code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
    return code

def tokenize(item):
    source, max_length, tokenizer = item
    source_tokens = [x for x in tokenizer.tokenize(source) if x!='\u0120']
    source_tokens = ["<s>","<decoder-only>","</s>"]+source_tokens[-(max_length-3):]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
    padding_length = max_length - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return source_tokens,source_ids

def convert_examples_to_features(examples, tokenizer, args,pool=None,stage=None):
    features = []
    if stage=="train":
        max_length = args.max_source_length+args.max_target_length
    else:
        max_length = args.max_source_length    
    sources = [(x.source,max_length,tokenizer) for x in examples]
    if pool is not None:
        tokenize_tokens = pool.map(tokenize,tqdm(sources,total=len(sources)))      
    else:
        tokenize_tokens = [tokenize(x) for x in sources]
    for example_index, (source_tokens,source_ids) in enumerate(tokenize_tokens):
        #source
        if example_index < 5:
            if stage=='train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example_index))

                logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
            )
        )
    return features



def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")   
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    parser.add_argument("--lang", default="python", type=str, 
                        help="Source language")  

    
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name") 
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    pool = multiprocessing.Pool(cpu_cont)
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU 
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count() 
    logger.warning("Process rank: %s, device: %s, n_gpu: %s",
                    args.local_rank, device, args.n_gpu)
    args.device = device
    
   
    # Set seed
    set_seed(args.seed)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    config =RobertaConfig.from_pretrained(args.model_name_or_path)
    config.is_decoder = True
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)

    #budild model
    encoder = RobertaModel.from_pretrained(args.model_name_or_path,config=config) 
    if args.lang == "python":
        eos_ids = [tokenizer.sep_token_id]
    else:
        eos_ids = [tokenizer.convert_tokens_to_ids('Ġ;'), tokenizer.convert_tokens_to_ids('Ġ}'), tokenizer.convert_tokens_to_ids('Ġ{')]
    
    model=Seq2Seq(encoder=encoder,decoder=encoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=eos_ids)
    
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
        
    model.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer,args, pool, stage='train')
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_source_ids)
        
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps)

        num_train_optimization_steps =  args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=int(len(train_dataloader)*args.num_train_epochs*0.1),
                                                    num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Step per epoch = %d", len(train_examples)//args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)
        

        model.train()
        dev_dataset={}
        nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_acc,best_loss = 0, 0,0,0,0,1e6 
        losses = []
        for epoch in range(args.num_train_epochs):

            for idx,batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                source_ids = batch[0]
                loss,_,_ = model(source_ids,True)
                
                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                losses.append(loss.item())
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
                if (idx+1)%100==0:
                    logger.info("epoch {} step {} loss {}".format(epoch,idx+1,round(np.mean(losses[-100:]),4)))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    #Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            if args.do_eval:
                #Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0                     
                eval_flag=False    

                eval_examples = read_examples(args.dev_filename.replace(".json",".txt"))
                eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='dev')
                all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)  
                eval_data = TensorDataset(all_source_ids)   
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
                
                model.eval() 
                dev_losses = []
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids = batch[0]
                    with torch.no_grad():
                        loss,_,_ = model(source_ids,True)
                        if args.n_gpu > 1:
                            loss = loss.mean() # mean() to average on multi-gpu.
                        dev_losses.append(loss.item())                            
                
                
                eval_examples = read_examples(args.dev_filename)
                eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='dev')
                all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)  
                eval_data = TensorDataset(all_source_ids)   
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
                p=[]
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids = batch[0]                  
                    with torch.no_grad():
                        preds = model(source_ids=source_ids)  
                        for pred in preds:
                            t=pred[0].cpu().numpy()
                            t=list(t)
                            if 0 in t:
                                t=t[:t.index(0)]
                            text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                            if args.lang == "java" and "{" in text:
                                text = text[:text.index("{")]                           
                            if args.lang == "python" and "</s>" in text:
                                text = text[:text.index("</s>")]
                            p.append(text)
                model.train()
                EM = 0.0
                edit_sim = 0.0    
                total = len(p)
                for ref,gold in zip(p,eval_examples):
                    pred = post_process(ref.strip())
                    gt = post_process(gold.target)
                    edit_sim += fuzz.ratio(pred, gt)
                    if pred.split() == gt.split():
                        EM += 1   
                dev_acc = round(EM/total*100, 2)
                logger.info("  %s = %s "%("loss",round(np.mean(dev_losses),4)))
                logger.info("  %s = %s "%("Acc",str(dev_acc)))
                logger.info("  %s = %s "%("Edit sim",str(round(edit_sim/total, 2))))
                logger.info("  "+"*"*20)    
                if dev_acc>best_acc:
                    logger.info("  Best acc:%s",dev_acc)
                    logger.info("  "+"*"*20)
                    best_acc=dev_acc
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
               
    if args.do_test:
        files=[]

        if args.test_filename is not None:
            files.append(args.test_filename)
        for idx,file in enumerate(files):   
            logger.info("Test file: {}".format(file))
            eval_examples = read_examples(file)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_source_ids)   

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval() 
            p=[]
            for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                source_ids = batch[0]                  
                with torch.no_grad():
                    preds = model(source_ids=source_ids)  
                    for pred in preds:
                        t=pred[0].cpu().numpy()
                        t=list(t)
                        if 0 in t:
                            t=t[:t.index(0)]
                        text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                        if args.lang == "java" and "{" in text:
                            text = text[:text.index("{")]
                        if args.lang == "python" and "</s>" in text:
                            text = text[:text.index("</s>")]
                        p.append(text)

            model.train()
            EM = 0.0
            edit_sim = 0.0 
            total = len(p)
            with open(os.path.join(args.output_dir, 'predictions.txt'),"w") as f:
                for ref,gold in zip(p,eval_examples):
                    pred = post_process(ref.strip())
                    gt = post_process(gold.target)
                    edit_sim += fuzz.ratio(pred, gt)
                    if pred.split() == gt.split():
                        EM += 1      
                    f.write(ref.strip()+"\n")


            dev_acc = round(EM/total*100, 2)
            logger.info("  %s = %s "%("Acc",str(dev_acc)))
            logger.info("  %s = %s "%("Edit sim",str(round(edit_sim/total, 2)))) 



                            

                
                
if __name__ == "__main__":
    main()



