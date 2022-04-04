# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, x):
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
    
        
    def forward(self, input_ids=None,labels=None): 
        input_ids = input_ids.view(-1,self.args.block_size)
        outputs = self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        outputs = (outputs * input_ids.ne(1)[:,:,None]).sum(1)/input_ids.ne(1).sum(1)[:,None]
        outputs = outputs.reshape(-1,2,outputs.size(-1))
        outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)
        cos_sim = (outputs[:,0]*outputs[:,1]).sum(-1)

        if labels is not None:
            loss = ((cos_sim-labels.float())**2).mean()
            return loss,cos_sim
        else:
            return cos_sim
      
        
 
        


