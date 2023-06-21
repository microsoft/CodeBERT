import random
import torch
from torch.utils.data import Dataset
import os
import pickle
import logging
import json
from tqdm import tqdm


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def _truncate_seq_pair_two_length(tokens_a, tokens_b, max_length_a, max_length_b):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length_a + max_length_b:
            break
        if len(tokens_b) > max_length_b:
            tokens_b.pop()
        else: # len(tokens_a) > max_length_a
            tokens_a.pop()

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 trace_tokens

    ):
        self.code_tokens = code_tokens
        self.trace_tokens = trace_tokens
        
def convert_examples_to_features(item):
    # parsing
    js,tokenizer=item
    code_tokens = js["code_tokens"]
    trace_tokens = js["trace_tokens"]
    code_tokens = tokenizer.tokenize(" ".join(code_tokens))
    trace_tokens = tokenizer.tokenize(" ".join(trace_tokens))

    return InputFeatures(code_tokens,trace_tokens)



class TextDataset(Dataset):
    def __init__(self, tokenizer, args, filename, local_rank, world_size, logger, mode, prefix=""):
        self.args = args
        self.tokenizer = tokenizer
        if len(prefix) > 0:
            cached_features_file = os.path.join('{}'.format(args.data_cache_dir), prefix + "_word_size_"+str(world_size)+"_rank_"+str(local_rank)+'_size_'+ str(args.block_size)+'_'+mode+'.pkl')
        else:
            cached_features_file = os.path.join('{}'.format(args.data_cache_dir), "word_size_"+str(world_size)+"_rank_"+str(local_rank)+'_size_'+ str(args.block_size)+'_'+mode+'.pkl')
        if os.path.exists(cached_features_file):
            logger.warning("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle1:
                self.examples = pickle.load(handle1)
            if 'train' in mode and local_rank==0:
                for idx, example in enumerate(self.examples[:1]):
                        logger.warning("*** Example ***")
                        logger.warning("idx: %s",idx)
                        logger.warning("code_tokens: {}".format(' '.join(map(str, example.code_tokens))))   
                        logger.warning("trace_tokens: {}".format(' '.join(map(str, example.trace_tokens))))           
        else:
            self.examples = []
            total_num = 0
            error_num = 0
            logger.info("Load and create features from dataset file at %s", filename)
            num_lines = sum(1 for line in open(filename,'r'))
            with open(filename,"r",encoding="utf8") as f:
                for i,line in enumerate(tqdm(f,total=num_lines)):
                    json_line = json.loads(line)
                    if len(json_line['code_tokens']) != 0: 
                        total_num += 1
                        if (mode == "train" and total_num % world_size == local_rank) or (mode != "train" and local_rank in [-1, 0]):
                            js = {}
                            if len(prefix) > 0:
                                js["code_tokens"] = ["<"+prefix+">"]
                                js["code_tokens"].extend(json_line["code_tokens"])
                            else:
                                js["code_tokens"] = json_line["code_tokens"]
                            js["trace_tokens"] = json_line["trace_tokens"]
                            try:
                                features = convert_examples_to_features((js, tokenizer))
                                cur_index = len(self.examples)
                                self.examples.append(features)
                            except:
                                error_num += 1 

            if mode == "train" and local_rank==0:
                for idx, example in enumerate(self.examples[:1]):
                    logger.warning("*** Example ***")
                    logger.warning("idx: %s",idx)
                    logger.warning("code_tokens: {}".format(example.code_tokens))   
                    logger.warning("trace_tokens: {}".format(example.trace_tokens))

            
            logger.warning("Num examples = %d: %d", local_rank,len(self.examples))
            logger.warning(f"Error num = {error_num}")
            # debug
            logger.warning("Saving features into cached file %s", cached_features_file)
            if not os.path.exists(args.data_cache_dir):
                os.makedirs(args.data_cache_dir)
            with open(cached_features_file, 'wb') as handle1:
                pickle.dump(self.examples, handle1, protocol=pickle.HIGHEST_PROTOCOL)
        
            


    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item): 
        js = self.examples[item]  

        # Encoder-Decoder for Trace Generation
        source_tokens = js.code_tokens
        target_tokens = ["<mask0>"] + js.trace_tokens    
        _truncate_seq_pair_two_length(source_tokens,target_tokens,self.args.block_size//4 - 1, self.args.block_size//2 + self.args.block_size//4 - 5)   
        source_tokens = source_tokens + ["<mask0>"]
        text_tokens = ["<s>","<encoder-decoder>","</s>"] + source_tokens + ["</s>"] + target_tokens + ["</s>"]
        text_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
        dual_gen_ids = text_ids + [self.tokenizer.pad_token_id]*(self.args.block_size-len(text_ids)) 
        dual_gen_type_ids = [1] * len(["<s>","<encoder-decoder>","</s>"] + source_tokens + ["</s>"]) + [2] * len(target_tokens + ["</s>"]) + [0]*(self.args.block_size-len(text_ids))        


        return (
               torch.tensor(dual_gen_ids),
               torch.tensor(dual_gen_type_ids),             
               )



            

    
