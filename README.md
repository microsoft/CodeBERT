# Code Pretraining Models

This repo contains code pretraining models in the CodeBERT series from Microsoft, including four models as of July 2022.
- CodeBERT (EMNLP 2020)
- GraphCodeBERT (ICLR 2021)
- UniXcoder (ACL 2022)
- CodeReviewer (ESEC/FSE 2022)

# CodeBERT

This repo provides the code for reproducing the experiments in [CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/pdf/2002.08155.pdf). CodeBERT is a pre-trained model for programming language, which is a multi-programming-lingual model pre-trained on NL-PL pairs in 6 programming languages (Python, Java, JavaScript, PHP, Ruby, Go). 

### Dependency

- pip install torch
- pip install transformers

### Quick Tour
We use huggingface/transformers framework to train the model. You can use our model like the pre-trained Roberta base. Now, We give an example on how to load the model.
```python
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.to(device)
```

### NL-PL Embeddings

Here, we give an example to obtain embedding from CodeBERT.

```python
>>> from transformers import AutoTokenizer, AutoModel
>>> import torch
>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
>>> model = AutoModel.from_pretrained("microsoft/codebert-base")
>>> nl_tokens=tokenizer.tokenize("return maximum value")
['return', 'Ġmaximum', 'Ġvalue']
>>> code_tokens=tokenizer.tokenize("def max(a,b): if a>b: return a else return b")
['def', 'Ġmax', '(', 'a', ',', 'b', '):', 'Ġif', 'Ġa', '>', 'b', ':', 'Ġreturn', 'Ġa', 'Ġelse', 'Ġreturn', 'Ġb']
>>> tokens=[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
['<s>', 'return', 'Ġmaximum', 'Ġvalue', '</s>', 'def', 'Ġmax', '(', 'a', ',', 'b', '):', 'Ġif', 'Ġa', '>', 'b', ':', 'Ġreturn', 'Ġa', 'Ġelse', 'Ġreturn', 'Ġb', '</s>']
>>> tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
[0, 30921, 4532, 923, 2, 9232, 19220, 1640, 102, 6, 428, 3256, 114, 10, 15698, 428, 35, 671, 10, 1493, 671, 741, 2]
>>> context_embeddings=model(torch.tensor(tokens_ids)[None,:])[0]
torch.Size([1, 23, 768])
tensor([[-0.1423,  0.3766,  0.0443,  ..., -0.2513, -0.3099,  0.3183],
        [-0.5739,  0.1333,  0.2314,  ..., -0.1240, -0.1219,  0.2033],
        [-0.1579,  0.1335,  0.0291,  ...,  0.2340, -0.8801,  0.6216],
        ...,
        [-0.4042,  0.2284,  0.5241,  ..., -0.2046, -0.2419,  0.7031],
        [-0.3894,  0.4603,  0.4797,  ..., -0.3335, -0.6049,  0.4730],
        [-0.1433,  0.3785,  0.0450,  ..., -0.2527, -0.3121,  0.3207]],
       grad_fn=<SelectBackward>)
```


### Probing

As stated in the paper, CodeBERT is not suitable for mask prediction task, while CodeBERT (MLM) is suitable for mask prediction task.


We give an example on how to use CodeBERT(MLM) for mask prediction task.
```python
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline

model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")

CODE = "if (x is not None) <mask> (x>1)"
fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)

outputs = fill_mask(CODE)
print(outputs)

```
Results
```python
'and', 'or', 'if', 'then', 'AND'
```
The detailed outputs are as follows:
```python
{'sequence': '<s> if (x is not None) and (x>1)</s>', 'score': 0.6049249172210693, 'token': 8}
{'sequence': '<s> if (x is not None) or (x>1)</s>', 'score': 0.30680200457572937, 'token': 50}
{'sequence': '<s> if (x is not None) if (x>1)</s>', 'score': 0.02133703976869583, 'token': 114}
{'sequence': '<s> if (x is not None) then (x>1)</s>', 'score': 0.018607674166560173, 'token': 172}
{'sequence': '<s> if (x is not None) AND (x>1)</s>', 'score': 0.007619690150022507, 'token': 4248}
```

### Downstream Tasks

For Code Search and Code Docsmentation Generation tasks, please refer to the [CodeBERT](https://github.com/guoday/CodeBERT/tree/master/CodeBERT) folder.



# GraphCodeBERT

This repo also provides the code for reproducing the experiments in [GraphCodeBERT: Pre-training Code Representations with Data Flow](https://openreview.net/pdf?id=jLoC4ez43PZ). GraphCodeBERT is a pre-trained model for programming language that considers the inherent structure of code i.e. data flow, which is a multi-programming-lingual model pre-trained on NL-PL pairs in 6 programming languages (Python, Java, JavaScript, PHP, Ruby, Go). 

For downstream tasks like code search, clone detection, code refinement and code translation, please refer to the [GraphCodeBERT](https://github.com/guoday/CodeBERT/tree/master/GraphCodeBERT) folder.

# UniXcoder

This repo will provide the code for reproducing the experiments in [UniXcoder: Unified Cross-Modal Pre-training for Code Representation](https://arxiv.org/pdf/2203.03850.pdf). UniXcoder is a unified cross-modal pre-trained model for programming languages to support both code-related understanding and generation tasks. 

Please refer to the [UniXcoder](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder) folder for tutorials and downstream tasks.

# CodeReviewer

This repo also provides the code for reproducing the experiments in [CodeReviewer: Pre-Training for Automating Code Review Activities](https://arxiv.org/abs/2203.09095). CodeReviewer is a model pre-trained with code change and code review data to support code review tasks.

Please refer to the [CodeReviewer](https://github.com/microsoft/CodeBERT/tree/master/CodeReviewer) folder for tutorials and downstream tasks.

## Contact

Feel free to contact Daya Guo (guody5@mail2.sysu.edu.cn), Shuai Lu (shuailu@microsoft.com) and Nan Duan (nanduan@microsoft.com) if you have any further questions.
