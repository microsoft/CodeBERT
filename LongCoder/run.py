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
import nltk
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from fuzzywuzzy import fuzz
from io import open
from itertools import cycle
import torch.nn as nn
import re
from model import Seq2Seq
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          LongformerConfig,  RobertaTokenizer)
from longcoder import LongcoderModel
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
from datasets import load_dataset
#load parsers
parsers={}

for lang in ['python','ruby','java','go','javascript','php','c','cpp','c_sharp']:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    if lang == "c_sharp":
        parsers["csharp"]= parser
    else:
        parsers[lang]= parser

def tokenize_code(parser,context):
    root_node = parser.parse(bytes(context,'utf8')).root_node
    tokens_index=tree_to_token_index(root_node)
    code=context.split('\n')
    code_tokens=[index_to_code_token(x,code) for x in tokens_index] 
    return " ".join(code_tokens)

def extract_global_statement(node,lang=None):
    indexs = []
    for child in node.children:
        indexs+=extract_global_statement(child,lang)
    if lang=="java":
        if any([x in node.type for x in ["class","package_declaration"]]):
            indexs.append(node.end_point[0]) 
        if any([x in node.type for x in ["method_declaration"]]): 
            indexs.append(node.start_point[0]) 
    elif lang=="python":
        if any([x in node.type for x in ["import_statement","function_definition","class_definition"]]): 
            indexs.append(node.start_point[0]) 
    elif lang=="csharp":
        if any([x in node.type for x in ["using_directive","class_declaration","method_declaration"]]): 
            indexs.append(node.start_point[0]) 
    return indexs

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



class TextDataset(Dataset):
    def __init__(self, tokenizer, args, split):
        self.examples = load_dataset(args.filename)[split]
        self.args = args
        self.tokenizer = tokenizer      
        self.split = split               
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        source = self.examples[i]["context"]
        source = re.sub("\n+", "\n", source)

        root_node = parsers[self.args.lang].parse(bytes(source,'utf8')).root_node
        try:
            statement_indexs = sorted(list(set(extract_global_statement(root_node,self.args.lang))))
        except:
            statement_indexs = [i for i in range(200)]

        target = self.examples[i]["gt"]
        source_tokens = self.tokenizer.tokenize(source.strip()) + ["Ċ"]
        statement_masks = []
        index = 0
        for x in source_tokens:
            if "Ċ" in x:
                if index in statement_indexs:
                    statement_masks.append(True)
                else:
                    statement_masks.append(False)
                index += 1
            else:
                statement_masks.append(False)
        if self.split == "train":
            max_length = self.args.max_source_length + self.args.max_target_length
            if len(source_tokens) <= max_length-3:
                index = 0
            else: 
                index = random.randint((len(source_tokens)-(max_length-3))//2,len(source_tokens)-(max_length-3))
            source_tokens = source_tokens[index:index+max_length-3]
            statement_masks = statement_masks[index:index+max_length-3]
        else:
            max_length = self.args.max_source_length
            source_tokens = source_tokens[-(max_length-3):]
            statement_masks = statement_masks[-(max_length-3):]


        source_tokens = ["<s>","<decoder-only>","</s>"]+source_tokens
        statement_masks = [False,False,False] + statement_masks
        cont = 0
        global_mask = []

        for  x in statement_masks:
            if len(source_tokens) - len(global_mask) < self.args.window_size:
                global_mask.append(False)
            elif x is False:
                global_mask.append(False)
            elif cont == self.args.max_global_length:
                global_mask.append(False)
            else:
                global_mask.append(True)
                cont += 1
        for i in range(len(source_tokens)):
            if cont == self.args.max_global_length:
                continue
            elif source_tokens[i] == "Ċ" and global_mask[i] is False:
                global_mask[i] = True
                cont += 1

        if sum(global_mask) == 0:
            global_mask[0] = True

        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)
        source_ids += [self.tokenizer.pad_token_id]* (max_length - len(source_ids))

        global_mask  = global_mask + [False] * (max_length - len(global_mask))
        return torch.tensor(source_ids),torch.tensor(global_mask)

def train(args, model, tokenizer):
    """ Train the model """
    #get training dataset
    train_dataset = TextDataset(tokenizer, args,"train")
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * args.num_train_epochs)
    

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Optimization steps per epoch = %d", len(train_dataloader))
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    
    model.train()
    losses, best_acc = [], 0.0 
    for idx in range(args.num_train_epochs): 
        losses = []
        for step,batch in enumerate(train_dataloader):
            #get inputs
            source_ids,global_mask = [x.to(args.device) for x in batch]
            # forward and return loss
            loss = model(source_ids,global_mask,True)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            #report loss
            losses.append(loss.item())
            if len(losses)%100 == 0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(np.mean(losses[-100:]),5)))
            
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 

  
        #evaluate    
        results = evaluate(args, model, tokenizer,"validation",  output_file_prefix = "dev")
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))    
            
        #save best model
        if results['eval_acc']>best_acc:
            best_acc = results['eval_acc']
            logger.info("  "+"*"*20)  
            logger.info("  Best acc:%s",best_acc)
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-acc'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer,split, test_number = 100000000, output_file_prefix = "test"):
    test_dataset = TextDataset(tokenizer, args, split)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size,num_workers=4)

    
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    losses = [] 
    gts = []
    preds = []

    for batch in test_dataloader:  
        source_ids,global_mask = [x.to(args.device) for x in batch]
        with torch.no_grad():
            loss = model(source_ids,global_mask,True)
            if args.n_gpu > 1:
                loss = loss.mean() 
            losses.append(loss.item())

            outs = model(source_ids,global_mask)  
            for pred in outs:
                t=pred[0].cpu().numpy()
                t=list(t)
                if 0 in t:
                    t=t[:t.index(0)]
                text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                if  "\n" in text:
                    text = text[:text.index("\n")]
                preds.append(text)
        if len(preds) >= test_number:
            break
    preds = preds[:test_number]
    gts = [test_dataset.examples[i]['gt'] for i in range(min(len(test_dataset.examples),test_number))]
    with open(os.path.join(args.output_dir,"{}.output".format(output_file_prefix)),'w') as f, open(os.path.join(args.output_dir,"{}.gold".format(output_file_prefix)),'w') as f1:
        for ref,gold in zip(preds,gts):
            f.write(ref+'\n')
            f1.write(gold+'\n')
        f.close()
        f1.close()     

    EM = []
    edit_sim = []    
    for ref,gold in zip(preds,gts):
        pred = tokenize_code(parser,ref)
        gt = tokenize_code(parser,gold)
        edit_sim.append(fuzz.ratio(pred, gt))
        EM.append(pred.split() == gt.split())  

    model.train()
    result = {
        "eval_loss":round(np.mean(losses),4),
        "eval_acc":round(np.mean(EM)*100,2),
        "eval_edit_sim":round(np.mean(edit_sim),2),
    }

    return result


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
    parser.add_argument("--lang", default=None, type=str,)    
    ## Other parameters
    parser.add_argument("--filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")


    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_global_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--window_size", default=512, type=int)

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

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
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count() 
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)
    args.device = device
    
   
    # Set seed
    set_seed(args.seed)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
        


    config = LongformerConfig.from_pretrained(args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config.attention_window = [args.window_size for x in config.attention_window]
    config.is_decoder_only = True
    #budild model
    encoder = LongcoderModel.from_pretrained(args.model_name_or_path,config=config)  

    eos_ids = [tokenizer.convert_tokens_to_ids('Ċ')]
    
    model=Seq2Seq(encoder=encoder,decoder=encoder,config=config,tokenizer=tokenizer,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=eos_ids)

    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
        
    model.to(device)

    if args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        train(args, model, tokenizer)
       
    if args.do_test:
        results = evaluate(args, model, tokenizer,"test", test_number = 10000000, output_file_prefix = "test")
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))    


                            

                
                
if __name__ == "__main__":
    main()


