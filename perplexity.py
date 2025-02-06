import sys

import kenlm
import pandas as pd
import sentencepiece

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sentence_splitter import SentenceSplitter
from cc_net import text_normalizer

sp = sentencepiece.SentencePieceProcessor()
sp.load(str("lm/en.sp.model"))    
model = kenlm.Model("lm/en.arpa.bin")

percentile_head = 30
percentile_tail = 60
cutoffs = pd.read_csv("lm/cutoff.csv", index_col=0)
cutoffs = {l: (cutoffs[l][percentile_head], cutoffs[l][percentile_tail]) for l in cutoffs.columns}

def pp(log_score, length):
    return 10.0 ** (-log_score / length)

def perplexity(content):
    #The lower the perplexity, the closer the data is to the targeted domain.
    clean_content = text_normalizer.normalize(content)
    clean_content = sp.encode_as_pieces(clean_content)
    clean_content = " ".join(clean_content)
   
    lines = clean_content.split('\n')
    doc_log_score, doc_length = 0, 0
    sentences = []
    for line in lines:       
        log_score = model.score(line)
        length = len(line.split()) + 1
        doc_log_score += log_score
        doc_length += length
        sentences.append(f"{pp(log_score, length)}\t{line}")

    #print("\n".join(sentences)+"\n")
    doc_score = round(pp(doc_log_score, doc_length), 1)
    return doc_score

def category(perplexity):
    lang = "en"
    if perplexity < 0:
        return "all"
    pp_head, pp_tail = cutoffs[lang]
    if perplexity < pp_head:
        return "high"
    if perplexity < pp_tail:
        return "middle"
    return "low"

if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print("Usage: python perplexity FILENAME")
        exit(1)
    
    target = sys.argv[1]

    with open(target) as f:
        text = f.read()
        p = perplexity(text)
        print(p)
    


