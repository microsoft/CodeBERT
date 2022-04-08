# UniXcoder

This repo will provide the code for reproducing the experiments in [UniXcoder: Unified Cross-Modal Pre-training for Code Representation](https://arxiv.org/pdf/2203.03850.pdf). UniXcoder is a unified cross-modal pre-trained model for programming languages to support both code-related understanding and generation tasks. 

Here, we provide three types of UniXcoder:

[unixcoder-base-unimodal](https://huggingface.co/microsoft/unixcoder-base-unimodal): Pre-trained on C4 and CodeSearchNet dataset (without NL)

[unixcoder-base](https://huggingface.co/microsoft/unixcoder-base): Continue pre-training ```unixcoder-base-unimodal``` on NL-PL pairs of CodeSearchNet dataset. The model can support six languages: **java, ruby, python, php, javascript, and go**. This model is reported in the paper.

[unixcoder-base-nine](https://huggingface.co/microsoft/unixcoder-base-nine):  Continue pre-training ```unixcoder-base-unimodal``` on NL-PL pairs of CodeSearchNet dataset and additional 1.5M NL-PL pairs of C, C++ and C# programming language. The model can support nine languages: **java, ruby, python, php, javascript, go, c, c++ and c#**.

## 1. Dependency

- pip install torch
- pip install transformers

## 2. Quick Tour
We implement a class to use UniXcoder and you can follow the code to build UniXcoder.
You can download the class by
```shell
wget https://raw.githubusercontent.com/microsoft/CodeBERT/master/UniXcoder/unixcoder.py
```

```python
import torch
from unixcoder import UniXcoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniXcoder("microsoft/unixcoder-base")
model.to(device)
```

In the following, we will give zero-shot examples for several tasks under different mode, including **code search (encoder-only)**, **code completion (decoder-only)**, **function name prediction (encoder-decoder)** , **API recommendation (encoder-decoder)**, **code summarization (encoder-decoder)**.

## 3. Encoder-only Mode

For encoder-only mode, we give an example of **code search**.

### 1) Code and NL Embeddings

Here, we give an example to obtain code fragment embedding from CodeBERT.

```python
# Encode maximum function
func = "def f(a,b): if a>b: return a else return b"
tokens_ids = model.tokenize([func],max_length=512,mode="<encoder-only>")
source_ids = torch.tensor(tokens_ids).to(device)
tokens_embeddings,max_func_embedding = model(source_ids)

# Encode minimum function
func = "def f(a,b): if a<b: return a else return b"
tokens_ids = model.tokenize([func],max_length=512,mode="<encoder-only>")
source_ids = torch.tensor(tokens_ids).to(device)
tokens_embeddings,min_func_embedding = model(source_ids)

# Encode NL
nl = "return maximum value"
tokens_ids = model.tokenize([nl],max_length=512,mode="<encoder-only>")
source_ids = torch.tensor(tokens_ids).to(device)
tokens_embeddings,nl_embedding = model(source_ids)

print(max_func_embedding.shape)
print(max_func_embedding)
```

```python
torch.Size([1, 768])
tensor([[ 8.6533e-01, -1.9796e+00, -8.6849e-01,  4.2652e-01, -5.3696e-01,
         -1.5521e-01,  5.3770e-01,  3.4199e-01,  3.6305e-01, -3.9391e-01,
         -1.1816e+00,  2.6010e+00, -7.7133e-01,  1.8441e+00,  2.3645e+00,
				 ...,
         -2.9188e+00,  1.2555e+00, -1.9953e+00, -1.9795e+00,  1.7279e+00,
          6.4590e-01, -5.2769e-02,  2.4965e-01,  2.3962e-02,  5.9996e-02,
          2.5659e+00,  3.6533e+00,  2.0301e+00]], device='cuda:0',
       grad_fn=<DivBackward0>)
```

### 2) Similarity between code and NL

Now, we calculate cosine similarity between NL and two functions. Although the difference of two functions is only a operator (```<``` and ```>```), UniXcoder can distinguish them.

```python
# Normalize embedding
norm_max_func_embedding = torch.nn.functional.normalize(max_func_embedding, p=2, dim=1)
norm_min_func_embedding = torch.nn.functional.normalize(min_func_embedding, p=2, dim=1)
norm_nl_embedding = torch.nn.functional.normalize(nl_embedding, p=2, dim=1)

max_func_nl_similarity = torch.einsum("ac,bc->ab",norm_max_func_embedding,norm_nl_embedding)
min_func_nl_similarity = torch.einsum("ac,bc->ab",norm_min_func_embedding,norm_nl_embedding)

print(max_func_nl_similarity)
print(min_func_nl_similarity)
```

```python
tensor([[0.3002]], device='cuda:0', grad_fn=<ViewBackward>)
tensor([[0.1881]], device='cuda:0', grad_fn=<ViewBackward>)
```

## 3. Decoder-only Mode

For decoder-only mode, we give an example of **code completion**.

```python
context = """
def f(data,file_path):
    # write json data into file_path in python language
"""
tokens_ids = model.tokenize([context],max_length=512,mode="<decoder-only>")
source_ids = torch.tensor(tokens_ids).to(device)
prediction_ids = model.generate(source_ids, decoder_only=True, beam_size=3, max_length=128)
predictions = model.decode(prediction_ids)
print(context+predictions[0][0])
```

```python
def f(data,file_path):
    # write json data into file_path in python language
    data = json.dumps(data)
    with open(file_path, 'w') as f:
        f.write(data)
```

## 4. Encoder-Decoder Mode

For encoder-decoder mode, we give two examples including: **function name prediction**, **API recommendation**, **code summarization**.

### 1) **Function Name Prediction**

```python
context = """
def <mask0>(data,file_path):
    data = json.dumps(data)
    with open(file_path, 'w') as f:
        f.write(data)
"""
tokens_ids = model.tokenize([context],max_length=512,mode="<encoder-decoder>")
source_ids = torch.tensor(tokens_ids).to(device)
prediction_ids = model.generate(source_ids, decoder_only=False, beam_size=3, max_length=128)
predictions = model.decode(prediction_ids)
print([x.replace("<mask0>","").strip() for x in predictions[0]])
```

```python
['write_json', 'write_file', 'to_json']
```

### 2) API Recommendation

```python
context = """
def write_json(data,file_path):
    data = <mask0>(data)
    with open(file_path, 'w') as f:
        f.write(data)
"""
tokens_ids = model.tokenize([context],max_length=512,mode="<encoder-decoder>")
source_ids = torch.tensor(tokens_ids).to(device)
prediction_ids = model.generate(source_ids, decoder_only=False, beam_size=3, max_length=128)
predictions = model.decode(prediction_ids)
print([x.replace("<mask0>","").strip() for x in predictions[0]])
```

```python
['json.dumps', 'json.loads', 'str']
```

### 3) Code Summarization

```python
context = """
# <mask0>
def write_json(data,file_path):
    data = json.dumps(data)
    with open(file_path, 'w') as f:
        f.write(data)
"""
tokens_ids = model.tokenize([context],max_length=512,mode="<encoder-decoder>")
source_ids = torch.tensor(tokens_ids).to(device)
prediction_ids = model.generate(source_ids, decoder_only=False, beam_size=3, max_length=128)
predictions = model.decode(prediction_ids)
print([x.replace("<mask0>","").strip() for x in predictions[0]])
```

```python
['Write JSON to file', 'Write json to file', 'Write a json file']
```

## 5.  Fine-tuning

For downstream tasks reported in the paper, please refer to the [downstream-tasks](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder/downstream-tasks) folders.



# Reference
If you use this code or UniXcoder, please consider citing us.

<pre><code>@article{guo2022unixcoder,
  title={UniXcoder: Unified Cross-Modal Pre-training for Code Representation},
  author={Guo, Daya and Lu, Shuai and Duan, Nan and Wang, Yanlin and Zhou, Ming and Yin, Jian},
  journal={arXiv preprint arXiv:2203.03850},
  year={2022}
}</code></pre>



