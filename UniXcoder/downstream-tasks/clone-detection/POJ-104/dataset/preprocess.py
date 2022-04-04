# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import json
from tqdm import tqdm
def files(path):
    g = os.walk(path) 
    file=[]
    for path,dir_list,file_list in g:  
        for file_name in file_list:  
            file.append(os.path.join(path, file_name))
    return file

cont=0
with open("train.jsonl",'w') as f:
    for i in tqdm(range(1,65),total=64):
        items=files("ProgramData/{}".format(i))
        for item in items:
            js={}
            js['label']=item.split('/')[1]
            js['index']=str(cont)
            js['code']=open(item,encoding='latin-1').read()
            f.write(json.dumps(js)+'\n')
            cont+=1
        
with open("valid.jsonl",'w') as f:
    for i in tqdm(range(65,81),total=16):
        items=files("ProgramData/{}".format(i))
        for item in items:
            js={}
            js['label']=item.split('/')[1]
            js['index']=str(cont)
            js['code']=open(item,encoding='latin-1').read()
            f.write(json.dumps(js)+'\n')
            cont+=1
            
with open("test.jsonl",'w') as f:
    for i in tqdm(range(81,195),total=24):
        items=files("ProgramData/{}".format(i))
        for item in items:
            js={}
            js['label']=item.split('/')[1]
            js['index']=str(cont)
            js['code']=open(item,encoding='latin-1').read()
            f.write(json.dumps(js)+'\n')
            cont+=1