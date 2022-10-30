# Testing Performance of Code Search using NL-PL Embbedding Models on CAT benchmark

This repo performs the evaluation of following NL-PL Embedding Models on code search: UnixCoder, GraphCodeBERT, CodeBERT, Roberta, BERT; on the test part of [CAT dataset]https://arxiv.org/abs/2207.05579. For UnixCoder, GraphCodeBERT and BERT, we use 2 approaches of extracting embedding for each comment and for each code snippets
- Approach 1 (_1): Get the [cls] embedding from sequence of words in comment/ code following [this tutorial](https://github.com/microsoft/CodeBERT/issues/112)
- Approach 2 (_2): Loading Models following nl-pl embedding tutorial of [UnixCoder](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder) .

There are 4 datasets provided in the CAT dataset: Funcom, CodeSearchNet, TLCodeSum (Java) and PCSD (Python). 
For each dataset, CAT provided 2 version: clean version and raw version. For each files in .json format, CAT's author provided an array of dictionaries. Each dictionary contained keys "comment" and "raw_code" to store the information of query and candidates for our code search function.
We report the accuracy as Mean Reciprocal Rank (MRR), BLEU-4 Cummulative, Top-1 to Top-5 accuracy at this location [link](https://drive.google.com/drive/folders/1dV9tDGiJkhXgs-58pdGs4mb0dwDTH4ZJ?usp=sharing).


## 1. How to run the test 
Run the following file:
``` python
python codeSearch.py
```

Input (default -can be changed in code at lines 195-197 ):

- {Your github repo location}/../{Your github repo location}_testCAT/input/*.json

This folder stored the json files can be downloaded from CAT dataset.

Output (default -can be changed in code at lines 195-197 ): 

- {Your github repo location}/../{Your github repo location}_testCAT/output/

We store the summary of results on different embedding models on a_summary files. 
You can find the details of the top-K for each query in [output/{jsonFileName}/{embeddingName}/accuracies/].


## 2. Algorithm
``` java
# dictOfPairs stored pairs of comment - code snippets for code comment and their embedding. 
function handleCodeSearch(dictOfPairs){
    listOfTopKs=[]
    listOfCandidates=getCands(dictOfPairs)
    
    for query in getQueries(dictOfPairs):
        dictCompareScoresForEachQuery={}
        for cand in listOfCandidates:
            distance=getEuclidDistance(query.emb,cand.emb)
            if not distance in dictCompareScoresForEachQuery.keys():
                dictCompareScoresForEachQuery[distance]=[]
            dictCompareScoresForEachQuery[distance].append(cand)
        sortByKeys(dictCompareScoresForEachQuery)
        topKForQuery=getTopK(query,dictCompareScoresForEachQuery)
        listOfTopKs.append(topKForQuery)
    
    mrrScore=getMRR(listOfTopKs)
    return mrrScore

        
}
```

## 2. Modifying default configuration:

- Modifying search size: using variable 'searchSize'. Set the searchSize larger than the size of the json file if you want to do code search on the whole array of the json file.
- Modifying cachine: using variable 'isUsingCache'. If you don't want to regenerate the vector each time running, set this variable to 'true'. We stored the cached embedding for queries and candidates for each models at output/cached_vectors folders.
- Modifying the vectorization: change the file 'UtilVectorGenerationTechniques.py'

# 3. Result
We achieved the following results:

1) 7584 pairs of comment-code in the clean version of tlcodesum:
``` python
Embedding Model	Embedding Size (default)	MRR
unixcoder_2	768	45.91%
unixcoder_1	768	36.52%
graphcodebert_2	768	8.10%
graphcodebert_1	768	4.18%
codebert_2	768	0.27%
codebert_1	768	0.60%
roberta_1	1024	0.27%
bert_1	768	0.41%
```


2) 8714 pairs of comment-code in the raw version of tlcodesum:
``` python
Embedding Model	Embedding Size (default)	MRR
unixcoder_2	768	38.51%
unixcoder_1	768	35.51%
graphcodebert_2	768	7.15%
graphcodebert_1	768	1.66%
codebert_2	768	0.32%
codebert_1	768	0.84%
roberta_1	1,024	0.21%
bert_1	768	0.43%
```

Details of results can be seen at [here](https://docs.google.com/spreadsheets/d/1GotMv6QtLl3_53bPbv9GcR_XVrvJcstVeHCyFEfhack/edit#gid=598477646)