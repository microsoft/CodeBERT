from UtilFunctions import *

import torch

def getEmbeddingFromModel(strInput,dictConfig):
    isOK = False
    lstVector = []
    modelName=dictConfig['modelName']
    device=dictConfig['device']
    if modelName=='unixcoder_1' or modelName=='graphcodebert_1' or modelName=='codebert_1' or modelName=='roberta_1' or modelName=='bert_1':
        # modelEmb, tokenizerEmb, maxTokenLength
        modelEmb=dictConfig['modelEmb']
        tokenizerEmb=dictConfig['tokenizerEmb']
        maxTokenLength = dictConfig['maxTokenLength']
        lstVector,isOK=getEmbeddingByUnixcoderOrGCBOrCB_solution1(strInput,modelEmb,tokenizerEmb,maxTokenLength)
        pass
    elif modelName=='unixcoder_2' or modelName=='graphcodebert_2' or modelName=='codebert_2' or modelName=='roberta_2'or modelName=='bert_2':
        modelEmb = dictConfig['modelEmb']
        maxTokenLength = dictConfig['maxTokenLength']
        lstVector,isOK=getEmbeddingByUnixcoderOrGCBOrCB_solution2(strInput, modelEmb, maxTokenLength,device)
        pass
    elif modelName=='fasttext':
        modelEmb = dictConfig['modelEmb']
        maxTokenLength = dictConfig['maxTokenLength']
        lstVector, isOK = getEmbeddingByFastText(strInput,modelEmb,maxTokenLength)
        pass
    return lstVector,isOK

# Solution 1: We follow this article to extract the [cls] embedding for the summarization of a comment/ code snippet
# https://github.com/microsoft/CodeBERT/issues/112
def getEmbeddingByUnixcoderOrGCBOrCB_solution1(strInput,modelEmb,tokenizerEmb,maxTokenLength):
    isOK = False
    lstVector=[]
    try:
        tokens = tokenizerEmb.tokenize(strInput)
        if len(tokens) > maxTokenLength:
            tokens=tokens[:maxTokenLength]
            # print(tokens)
            # input('aaaa')
        tokens = [tokenizerEmb.cls_token] + tokens + [tokenizerEmb.sep_token]

        tokens_ids = tokenizerEmb.convert_tokens_to_ids(tokens)
        context_embeddings = modelEmb(torch.tensor(tokens_ids)[None, :])[0][0, 0]
        lstVector = context_embeddings.tolist()
        if len(lstVector)>0:
            isOK=True
    except Exception as e:
        traceback.print_exc()
        print(strInput)
        # input('aaa ')
        pass
    # print('len vector {}'.format(len(lstVector)))
    return lstVector,isOK

# Solution 2: We use this article to implement the embedding extraction:
# https://github.com/microsoft/CodeBERT/tree/master/UniXcoder
def getEmbeddingByUnixcoderOrGCBOrCB_solution2(strInput,modelEmb,maxTokenLength,device):
    isOK = False
    lstVector=[]
    try:
        tokens_ids = modelEmb.tokenize([strInput], max_length=maxTokenLength, mode="<encoder-only>")
        source_ids = torch.tensor(tokens_ids).to(device)
        tokens_embeddings, context_embedding = modelEmb(source_ids)
        lstVector = context_embedding.tolist()[0]

        if len(lstVector)>0:
            isOK=True
    except Exception as e:
        traceback.print_exc()
        print(strInput)
        input('aaa ')
        pass
    # print('len vector {}'.format(len(lstVector)))
    return lstVector,isOK


def getEmbeddingByFastText(strInput,modelEmb,maxTokenLength):
    isOK = False
    lstVector=[]
    try:
        tokens = strInput.split()
        strValue=strInput
        if len(tokens) > maxTokenLength:
            tokens=tokens[:maxTokenLength]
            # print(tokens)
            # input('aaaa')
            strValue=' '.join(strValue)
        lstVector = modelEmb.get_sentence_vector(strValue).tolist()[0]
        if len(lstVector)>0:
            isOK=True
    except Exception as e:
        traceback.print_exc()
        print(strInput)
        # input('aaa ')
        pass
    return lstVector,isOK
