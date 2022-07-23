import os
import argparse
from evaluator.smooth_bleu import bleu_fromstr
import nltk
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()
    ref = os.path.join(args.path, 'golds.txt')
    hyp = os.path.join(args.path, 'preds.txt')
    with open(ref, 'r') as f:
        refs = f.readlines()
    with open(hyp, 'r') as f:
        hyps = f.readlines()
    # refs = [ref.strip().lower() for ref in refs]
    # hyps = [hyp.strip().lower() for hyp in hyps]
    # bleu = bleu_fromstr(hyps, refs)
    # print(bleu)
    pred_nls, golds = hyps, refs
    for i in range(len(pred_nls)):
        chars = "(_)`."
        for c in chars:
            pred_nls[i] = pred_nls[i].replace(c, " " + c + " ")
            pred_nls[i] = " ".join(pred_nls[i].split())
            golds[i] = golds[i].replace(c, " " + c + " ")
            golds[i] = " ".join(golds[i].split())
    bleu = bleu_fromstr(pred_nls, golds, rmstop=False)
    print(bleu)
    # stopwords = open("stopwords.txt").readlines()
    # stopwords = [stopword.strip() for stopword in stopwords]
    # refs = [" ".join([word for word in ref.lower().split() if word not in stopwords]) for ref in refs]
    # hyps = [" ".join([word for word in hyp.lower().split() if word not in stopwords]) for hyp in hyps]
    # bleu = bleu_fromstr(hyps, refs)
    # print(bleu)

if __name__ == '__main__':
    main()
    # s = "Can we use `mset.mirrorInfo()` directly?"
    # chars = "(_)`."
    # for c in chars:
    #     s = s.replace(c, " " + c + " ")
    # print(nltk.wordpunct_tokenize(s))
