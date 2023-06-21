import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import random

    
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.lm_head = nn.Linear(config.hidden_size,config.vocab_size)
        self.qa_outputs = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        self.register_buffer(
        "bias", torch.tril(torch.ones((args.block_size, args.block_size), dtype=torch.uint8)).view(1, args.block_size, args.block_size)
        )
        self.weights = torch.full([len(self.tokenizer)], 10.0).to(self.args.device)
        easy_ids = self.tokenizer.convert_tokens_to_ids(["<state>","</state>", "<dictsep>", ":"])
        for i in easy_ids: self.weights[i] = 1.0

    def forward(self, dual_gen_ids, dual_gen_type_ids): 
        dual_loss,align_loss,contras_loss = 0,0,0
        
        # Encoder-Decoder for Cross-modal Generation
        source_ids = dual_gen_ids
        type_ids = dual_gen_type_ids
        attention_mask = self.bias
        attention_mask = attention_mask | (type_ids.eq(1)[:,:,None]*type_ids.eq(1)[:,None,:])
        outputs = self.encoder(source_ids,attention_mask=attention_mask)
        encoder_outputs = outputs.last_hidden_state[:,:-1]
        labels_mask = type_ids.eq(2)[:,1:]
        encoder_outputs = encoder_outputs.reshape(-1,encoder_outputs.size(-1))[labels_mask.reshape(-1)]
        prediction_scores = self.lm_head(encoder_outputs)
        lm_labels = source_ids[:,1:].reshape(-1)[labels_mask.reshape(-1)]

        loss_fct = CrossEntropyLoss(reduction='none')
        lm_loss = loss_fct(prediction_scores, lm_labels)
        lm_loss = self.weights[lm_labels] * lm_loss
        lm_loss = lm_loss.sum()/len(lm_labels)

        dual_loss = lm_loss.item() 
        return lm_loss, dual_loss, align_loss, contras_loss



