import ast
import json
import shutil
import traceback
from statistics import mean,median
import pickle
from os.path import exists
import glob
from UtilFunctions import *
from UtilVectorGenerationTechniques import *
from scipy.spatial import distance
from nltk.translate.bleu_score import sentence_bleu
from transformers import BertTokenizer, BertModel,RobertaModel,RobertaTokenizer
from transformers import AutoTokenizer, AutoModel

def checkSetOverlap(setCorrect,setCandSameScore):
    isFound=False
    for itemCorrect in setCorrect:
        if itemCorrect in setCandSameScore:
            isFound=True
            break
    return isFound

def handleQuery(dictQueriesAndCandidates,dictConfig):
    lstTopKAllOrigin = []
    lstTopKAllAugment = []
    lstBLEUAllOrigin = []
    lstBLEUAllAugment = []
    dictResult={}
    lstKeyAll=list(dictQueriesAndCandidates.keys())
    fopOutput=dictConfig['fopOutput']
    # batchSizeOfResults=dictConfig['batchSizeOfResults']
    # vectorEngine=dictConfig['vectorEngine']

    fopOutputScore=fopOutput+'accuracy/'
    createDirIfNotExist(fopOutputScore)
    fopOutputDetails = fopOutput + 'cached_print/'
    createDirIfNotExist(fopOutputDetails)
    fpDetailOriginal=fopOutputDetails+'org.txt'
    fpDetailAugmentation = fopOutputDetails + 'aug.txt'
    f1 = open(fpDetailOriginal, 'w')
    f1.write('Key\tTopK-org\tDictScoreOriginal\n')
    f1.close()


    indexQuery=-1
    for keyQuery in lstKeyAll:
        try:
            indexQuery+=1
            print('begin query {}'.format(indexQuery))
            # jsonItemQueryAndCands = ast.literal_eval(arrItemBatchs[j])
            strQuery=dictQueriesAndCandidates[keyQuery]['queryText']
            # lstCandidatesBefore=jsonItemQueryAndCands['candidates']
            setIndexCorrects=set()
            setIndexCorrects.add(keyQuery)
            dictScoreAndOutputOrigin = {}
            # dictItemCandidateBefore={}
            dictScoreAndOutputAugment = {}
            lstKeyCandidates=lstKeyAll
            startIndex=0
            endIndex=len(lstKeyAll)


            dictCandKeysToOrder={}
            indexCandOrder=0
            # print('keyCand {}'.format(len(lstKeyCandidates)))
            for i in range(0,len(lstKeyCandidates)):
                keyCand=lstKeyCandidates[i]
                # print('keyCand {}'.format(keyCand))
                indexCandOrder+=1
                isCorrect=False
                if keyCand==keyQuery:
                    isCorrect=True
                    # print('go here')
                    # input('aaaa')
                dictCandKeysToOrder[keyCand]=[indexCandOrder,isCorrect]
            # print(dictCandKeysToOrder)
            # input('aaaa')
            indexCand=-1
            vectorQueryOrigin=dictQueriesAndCandidates[keyQuery]['queryEmbOriginal']
            for keyCand in lstKeyCandidates:
                itemCand=dictQueriesAndCandidates[keyCand]
                distanceJKOrigin = 0
                # print('vector query {}\n vector cand {}'.format(vectorQueryOrigin,itemCand['candEmbOriginal']))
                distanceJKOrigin=distance.euclidean(vectorQueryOrigin, itemCand['candEmbOriginal'])
                itemCand['distanceQueryCandidateOrigin']=distanceJKOrigin
                if not distanceJKOrigin in dictScoreAndOutputOrigin.keys():
                    dictScoreAndOutputOrigin[distanceJKOrigin] = set()
                dictScoreAndOutputOrigin[distanceJKOrigin].add(keyCand)

            dictScoreAndOutputOrigin = {k: dictScoreAndOutputOrigin[k] for k in sorted(dictScoreAndOutputOrigin)}
            topKForItemOrigin = 0
            lstKeyScores = list(dictScoreAndOutputOrigin.keys())
            for indexKeyScore in range(0, len(lstKeyScores)):
                setIndexesSameScore = dictScoreAndOutputOrigin[lstKeyScores[indexKeyScore]]
                itemIsFound=checkSetOverlap(setIndexCorrects, setIndexesSameScore)
                if itemIsFound:
                    topKForItemOrigin = topKForItemOrigin + 1
                    break
                else:
                    topKForItemOrigin += len(setIndexesSameScore)

            lstRankScoreKeyOrigin=[]
            indexRankScoreOrigin=0
            for keyScore in dictScoreAndOutputOrigin.keys():
                indexRankScoreOrigin+=1
                setCandSameScore=dictScoreAndOutputOrigin[keyScore]
                for keyCandSameScore in setCandSameScore:
                    lstSetOrderAndIsCorrect=dictCandKeysToOrder[keyCandSameScore]
                    strDisplay=lstSetOrderAndIsCorrect[0]
                    if lstSetOrderAndIsCorrect[1]:
                        strDisplay='{} (correct)'.format(lstSetOrderAndIsCorrect[0])
                    lstRankScoreKeyOrigin.append([indexRankScoreOrigin,keyScore,strDisplay])

            selectedBLEUScoreOrigin=0
            for idxCorrect in setIndexCorrects:
                firstKeyOrigin=list(dictScoreAndOutputOrigin.keys())[0]
                setOriginIndex=dictScoreAndOutputOrigin[firstKeyOrigin]
                arrTextCorrect=dictQueriesAndCandidates[idxCorrect]['candText'].split()
                for idxPreOrigin in setOriginIndex:
                    arrTextPreOrigin = dictQueriesAndCandidates[idxPreOrigin]['candText'].split()
                    bleuScoreItem = sentence_bleu([arrTextCorrect], arrTextPreOrigin,
                                                  weights=(0.25, 0.25, 0.25, 0.25))
                    if bleuScoreItem>selectedBLEUScoreOrigin:
                        selectedBLEUScoreOrigin=bleuScoreItem

            lstBLEUAllOrigin.append(selectedBLEUScoreOrigin)
            lstTopKAllOrigin.append(topKForItemOrigin)
            f1 = open(fpDetailOriginal, 'a')
            f1.write('{}\t{}\t{}\n'.format(keyQuery,topKForItemOrigin,str(dictScoreAndOutputOrigin)))
            f1.close()
            f1 = open(fpDetailAugmentation, 'a')
            f1.write('{}\t{}\t{}\n'.format(keyQuery,topKForItemOrigin,str(dictScoreAndOutputAugment)))
            f1.close()



        except Exception as e:
            traceback.print_exc()
            # input('aaa')
        # f1=open(fpCache,'a')
        # f1.write(str(dictStoreAllQueryResults)+'\n')
        # f1.close()
        if (indexQuery+1)%100==0:
            fpLogTopKAndBLEU = fopOutputScore + 'logTopKAndBLEU.txt'
            f1 = open(fpLogTopKAndBLEU, 'w')
            f1.write('Key\tTopK\tBLEU4\n')
            for i in range(0, indexQuery):
                f1.write('{}\t{}\t{}\n'.format(lstKeyAll[i], lstTopKAllOrigin[i],
                                                       lstBLEUAllOrigin[i]))
            f1.close()

            lstStr = ['Top-K,Number,Accuracy']
            for i in range(1, 6):
                numberTopI = sum(item <= i for item in lstTopKAllOrigin)
                percentageTopI = numberTopI / len(lstTopKAllOrigin)
                lstStr.append('Top-{},{}/{},{}'.format(i, numberTopI, len(lstTopKAllOrigin), percentageTopI))
            mrrScoreOrigin = mean([1 / i for i in lstTopKAllOrigin])
            lstStr.append('MRR: {}'.format(mrrScoreOrigin))
            avgBLEU = mean(lstBLEUAllOrigin)
            lstStr.append('BLEU: {}\n'.format(avgBLEU))

            fpSum = fopOutputScore + 'summaryPerModel.txt'
            f1 = open(fpSum, 'w')
            f1.write('\n'.join(lstStr))
            f1.close()

    fpLogTopKAndBLEU=fopOutputScore+'logTopKAndBLEU.txt'
    f1 = open(fpLogTopKAndBLEU, 'w')
    f1.write('Key\tTopK\tBLEU-4 (Cummulate)\n')
    for i in range(0, len(lstKeyAll)):
        f1.write('{}\t{}\t{}\n'.format(lstKeyAll[i], lstTopKAllOrigin[i],
                                               lstBLEUAllOrigin[i]))
    f1.close()

    lstStr = ['Top-K,Number,Accuracy']
    for i in range(1, 6):
        numberTopI = sum(item <= i for item in lstTopKAllOrigin)
        percentageTopI = numberTopI / len(lstTopKAllOrigin)
        dictResult['Top-{}'.format(i)] =percentageTopI
        lstStr.append('Top-{},{}/{},{}'.format(i, numberTopI, len(lstTopKAllOrigin), percentageTopI))
    mrrScoreOrigin = mean([1 / i for i in lstTopKAllOrigin])
    lstStr.append('MRR: {}'.format(mrrScoreOrigin))
    dictResult['MRR'] = mrrScoreOrigin
    avgBLEU = mean(lstBLEUAllOrigin)
    lstStr.append('BLEU: {}\n'.format(avgBLEU))
    dictResult['BLEU4'] = avgBLEU

    fpSum = fopOutputScore + 'summaryPerModel.txt'
    f1 = open(fpSum, 'w')
    f1.write('\n'.join(lstStr))
    f1.close()
    return dictResult

fopRoot='../../CodeBERT_testCAT/'
fopInput=fopRoot+'input_2/'
fopOutput=fopRoot+'output_2/'
createDirIfNotExist(fopOutput)

fopJsonInDataset=fopInput+'jsons/'
fpFasttext=fopInput+'modelFasttext/fasttext.cbow.100.bin'
searchSize=100000
batchSizeOfCaching=500
isUsingCache=True
maxTokenLength=500

# embedding models provided
lstEmbeddingEngines=[
    'unixcoder_2',
'unixcoder_1',
     'graphcodebert_2',
    'graphcodebert_1',
     'codebert_2',
        'codebert_1',
        'roberta_1',
            'bert_1'
]

from unixcoder import UniXcoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizerUnixcoder1 = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
modelUnixcoder1 = AutoModel.from_pretrained("microsoft/unixcoder-base")
tokenizerGraphCodeBERT1 = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
modelGraphCodeBERT1 = AutoModel.from_pretrained("microsoft/graphcodebert-base")
tokenizerCodeBERT1 = AutoTokenizer.from_pretrained("microsoft/codebert-base")
modelCodeBERT1 = AutoModel.from_pretrained("microsoft/codebert-base")
tokenizerRoberta1 = RobertaTokenizer.from_pretrained('roberta-large')
modelRoberta1 = RobertaModel.from_pretrained('roberta-large')
tokenizerBERT1 = BertTokenizer.from_pretrained('bert-base-uncased')
modelBERT1 = BertModel.from_pretrained('bert-base-uncased')


modelUnixCoder2 = UniXcoder("microsoft/unixcoder-base")
modelCodeBERT2 = UniXcoder("microsoft/codebert-base")
modelGraphCodeBERT2 = UniXcoder("microsoft/graphcodebert-base")
modelUnixCoder2.to(device)
modelCodeBERT2.to(device)
modelGraphCodeBERT2.to(device)

fpSummaryOverall= fopOutput + 'a_summary_all.txt'
f1=open(fpSummaryOverall,'w')
strHeadLine='Search Size\tCorpus\tEmbedding Model\tEmbedding Size (default)\tMRR\tBLEU-4 (Cummulated)\tTop-1\tTop-2\tTop-3\tTop-4\tTop-5'
f1.write(strHeadLine+'\n')
f1.close()
lstFpJsons=glob.glob(fopJsonInDataset+'*')

for indexFileJson in range(0,len(lstFpJsons)):
    currentFpItemFileJson=lstFpJsons[indexFileJson]
    nameOfJsonFile=os.path.basename(currentFpItemFileJson)
    fopItemOutputPerJsonFile=fopOutput+nameOfJsonFile+'/'
    createDirIfNotExist(fopItemOutputPerJsonFile)
    try:

        f1 = open(currentFpItemFileJson, 'r')
        arrItemJsons = f1.read().strip().split('\n')
        f1.close()

        dictJsonOriginalInfo = {}
        for j in range(0, len(arrItemJsons)):
            try:
                dictItemJson = ast.literal_eval(arrItemJsons[j])
                dictJsonOriginalInfo[j] = dictItemJson
            except Exception as e:
                traceback.print_exc()
        lstKeyItemJsonOriginal = list(dictJsonOriginalInfo.keys())

        for indexEmbModel in range(0, len(lstEmbeddingEngines)):
            # currentEmbModel=lstEmbeddingCouples[indexEmbModel]
            currentEmbModelName = lstEmbeddingEngines[indexEmbModel]
            currentBatchOfTestIndex = -1


            # dictJsonOriginalInfo=pickle.load(open(fpJsonInDataset,'rb'))
            dictQueryAndCandidates = {}
            fopItemOutputPerEmbModel=fopItemOutputPerJsonFile+currentEmbModelName+'/'
            createDirIfNotExist(fopItemOutputPerEmbModel)
            lenItemEmb = -1
            # sizeItemReduceEmb=-1
            fopItemOutputCachedVector=fopItemOutputPerEmbModel+'cached_vectors/'
            createDirIfNotExist(fopItemOutputCachedVector)
            lstElemenetsInFopCachedVector=os.listdir(fopItemOutputCachedVector)
            # if isUsingCache and
            dictConfigOfEmbeddingModel={}
            dictConfigOfEmbeddingModel['modelName']=currentEmbModelName
            dictConfigOfEmbeddingModel['maxTokenLength']=maxTokenLength
            dictConfigOfEmbeddingModel['device'] = device

            if currentEmbModelName=='unixcoder_2':
                dictConfigOfEmbeddingModel['modelEmb']=modelUnixCoder2
            elif currentEmbModelName=='graphcodebert_2':
                dictConfigOfEmbeddingModel['modelEmb'] = modelGraphCodeBERT2
            elif currentEmbModelName=='codebert_2':
                dictConfigOfEmbeddingModel['modelEmb'] = modelCodeBERT2
            elif currentEmbModelName=='unixcoder_1':
                dictConfigOfEmbeddingModel['modelEmb']=modelUnixcoder1
                dictConfigOfEmbeddingModel['tokenizerEmb'] = tokenizerUnixcoder1
            elif currentEmbModelName=='roberta_1':
                dictConfigOfEmbeddingModel['modelEmb'] = modelRoberta1
                dictConfigOfEmbeddingModel['tokenizerEmb'] = tokenizerRoberta1
            elif currentEmbModelName=='bert_1':
                dictConfigOfEmbeddingModel['modelEmb'] = modelBERT1
                dictConfigOfEmbeddingModel['tokenizerEmb'] = tokenizerBERT1
            elif currentEmbModelName=='codebert_1':
                dictConfigOfEmbeddingModel['modelEmb'] = modelCodeBERT1
                dictConfigOfEmbeddingModel['tokenizerEmb'] = tokenizerCodeBERT1
            elif currentEmbModelName=='graphcodebert_1':
                dictConfigOfEmbeddingModel['modelEmb'] = modelGraphCodeBERT1
                dictConfigOfEmbeddingModel['tokenizerEmb'] = tokenizerGraphCodeBERT1

            if not isUsingCache or len(glob.glob(fopItemOutputCachedVector+'/*.pkl'))==0:
                shutil.rmtree(fopItemOutputCachedVector, ignore_errors=False, onerror=None)
                createDirIfNotExist(fopItemOutputCachedVector)
                currentBatchOfTestIndex = -1
                dictCurrentBatchQuery={}
                for j in range(0, len(lstKeyItemJsonOriginal)):
                    try:

                        keyInJson = lstKeyItemJsonOriginal[j]
                        newBatchIndex = keyInJson // batchSizeOfCaching

                        if newBatchIndex != currentBatchOfTestIndex:
                            if currentBatchOfTestIndex != -1:
                                fpItemBatch = '{}/{}.pkl'.format(fopItemOutputCachedVector, currentBatchOfTestIndex)
                                pickle.dump(dictCurrentBatchQuery, open(fpItemBatch, 'wb'))
                            currentBatchOfTestIndex = newBatchIndex
                            fpItemBatch = '{}/{}.pkl'.format(fopItemOutputCachedVector, currentBatchOfTestIndex)
                            if os.path.exists(fpItemBatch):
                                dictCurrentBatchQuery = pickle.load(open(fpItemBatch, 'rb'))

                        dictItemLineOfInput = dictJsonOriginalInfo[keyInJson]
                        # print('key json {}'.format(keyInJson))

                        if newBatchIndex != currentBatchOfTestIndex:
                            fpPklBatch = fopItemOutputCachedVector + '{}.pkl'.format(newBatchIndex)
                            dictCurrentVector = pickle.load(open(fpPklBatch, 'rb'))
                            currentBatchOfTestIndex = newBatchIndex

                        dictItemParallels = {}
                        dictItemParallels['queryText'] = dictJsonOriginalInfo[keyInJson]['comment']
                        dictItemParallels['candText'] = dictJsonOriginalInfo[keyInJson]['raw_code']
                        lstVectorItemQuery,isOKQuery=getEmbeddingFromModel(dictItemParallels['queryText'] ,dictConfigOfEmbeddingModel)
                        lstVectorItemCand, isOKCand = getEmbeddingFromModel(dictItemParallels['candText'],
                                                                              dictConfigOfEmbeddingModel)
                        if isOKQuery and isOKCand:
                            dictItemParallels['queryEmbOriginal'] = lstVectorItemQuery
                            dictItemParallels['candEmbOriginal'] = lstVectorItemCand
                            dictQueryAndCandidates[keyInJson] = dictItemParallels
                            dictCurrentBatchQuery[keyInJson]=dictItemParallels
                            newBatchIndex=keyInJson//batchSizeOfCaching


                    except Exception as e:
                        traceback.print_exc()
                    if j==len(lstKeyItemJsonOriginal)-1:
                        fpItemBatch = '{}/{}.pkl'.format(fopItemOutputCachedVector, currentBatchOfTestIndex)
                        pickle.dump(dictCurrentBatchQuery, open(fpItemBatch, 'wb'))
            else:
                currentBatchOfTestIndex = -1
                dictCurrentBatchQuery = {}
                for j in range(0, len(lstKeyItemJsonOriginal)):
                    try:
                        keyInJson = lstKeyItemJsonOriginal[j]
                        newBatchIndex = keyInJson // batchSizeOfCaching

                        if newBatchIndex != currentBatchOfTestIndex:
                            if currentBatchOfTestIndex != -1:
                                fpItemBatch = '{}/{}.pkl'.format(fopItemOutputCachedVector, currentBatchOfTestIndex)
                                pickle.dump(dictCurrentBatchQuery, open(fpItemBatch, 'wb'))
                            currentBatchOfTestIndex = newBatchIndex
                            fpItemBatch = '{}/{}.pkl'.format(fopItemOutputCachedVector, currentBatchOfTestIndex)
                            if os.path.exists(fpItemBatch):
                                dictCurrentBatchQuery = pickle.load(open(fpItemBatch, 'rb'))
                        # print(dictCurrentBatchQuery)
                        if not keyInJson in dictCurrentBatchQuery.keys():
                            dictItemLineOfInput = dictJsonOriginalInfo[keyInJson]
                            # print('key json {}'.format(keyInJson))

                            dictItemParallels = {}
                            dictItemParallels['queryText'] = dictJsonOriginalInfo[keyInJson]['comment']
                            dictItemParallels['candText'] = dictJsonOriginalInfo[keyInJson]['raw_code']
                            lstVectorItemQuery, isOKQuery = getEmbeddingFromModel(dictItemParallels['queryText'],
                                                                                  dictConfigOfEmbeddingModel)
                            lstVectorItemCand, isOKCand = getEmbeddingFromModel(dictItemParallels['candText'],
                                                                                dictConfigOfEmbeddingModel)
                            # print('{}\n{}\n{}\n{}',isOKQuery,len(lstVectorItemQuery),isOKCand,len(lstVectorItemCand))
                            # input('bbbb')
                            if isOKQuery and isOKCand:
                                dictItemParallels['queryEmbOriginal'] = lstVectorItemQuery
                                dictItemParallels['candEmbOriginal'] = lstVectorItemCand
                                dictQueryAndCandidates[keyInJson] = dictItemParallels
                                dictCurrentBatchQuery[keyInJson] = dictItemParallels
                                # print('go here')
                        else:
                            dictItemParallels =dictCurrentBatchQuery[keyInJson]
                            dictQueryAndCandidates[keyInJson] = dictItemParallels
                            # print('{}', dictItemParallels)
                            # input('bbbb')
                    except Exception as e:
                        traceback.print_exc()
                    if j==len(lstKeyItemJsonOriginal)-1:
                        if currentBatchOfTestIndex!=-1:
                            fpItemBatch = '{}/{}.pkl'.format(fopItemOutputCachedVector, currentBatchOfTestIndex)
                            pickle.dump(dictCurrentBatchQuery, open(fpItemBatch, 'wb'))
                pass
            # using cache data

            listQueryKeys=list(dictQueryAndCandidates.keys())
            # print('before {}'.format(listQueryKeys))
            if searchSize>0 and searchSize<len(listQueryKeys):
                dictSmallerSize={}
                for q in range(0,min(searchSize,len(listQueryKeys))):
                    keyQ=listQueryKeys[q]
                    valQ=dictQueryAndCandidates[listQueryKeys[q]]
                    dictSmallerSize[keyQ]=valQ
                dictQueryAndCandidates=dictSmallerSize
            listQueryKeys = list(dictQueryAndCandidates.keys())
            # print('len {}'.format(len(listQueryKeys)))
            # input('hello world')
            embSizeDefault =0
            if len(listQueryKeys)>0:
                embSizeDefault=len(dictQueryAndCandidates[list(dictQueryAndCandidates.keys())[0]]['queryEmbOriginal'])
                # print('emd {}'.format(dictQueryAndCandidates[list(dictQueryAndCandidates.keys())[0]]['queryEmbOriginal']))
                # input('bbb')
            dictConfigOfSearchPhase={}
            # fopOutput = dictConfig['fopOutput']
            dictConfigOfSearchPhase['fopOutput']=fopItemOutputPerEmbModel
            embSizeDefault = len(list(dictQueryAndCandidates.keys()))
            dictResults = handleQuery(dictQueryAndCandidates, dictConfigOfSearchPhase)
            f1 = open(fpSummaryOverall, 'a')
            # strHeadLine = 'Search Size\tCorpus\tEmbedding Model\tEmbedding Size (default)\tMRR\tBLEU-4 (Cummulated)\tTop-1\tTop-2\tTop-3\tTop-4\tTop-5'
            strContentLine = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(searchSize,nameOfJsonFile,
                currentEmbModelName, embSizeDefault, dictResults['MRR'], dictResults['BLEU4'], dictResults['Top-1'],
                dictResults['Top-2'], dictResults['Top-3'], dictResults['Top-4'],
                dictResults['Top-5'])
            f1.write(strContentLine + '\n')
            f1.close()
            # except Exception as e:
            #     traceback.print_exc()
            #     # input('I want to sleep')
    except Exception as e:
        traceback.print_exc()












