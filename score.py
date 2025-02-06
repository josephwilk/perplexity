import sys
import glob, re

import finewebed
import perplexity

from concurrent.futures import ThreadPoolExecutor

if(len(sys.argv) < 2):
    print("Usage: python score.py DIRNAME")
    exit(1)
    
target = sys.argv[1]
DATA_FILES = glob.glob(target+'**/*.txt', recursive=True)

def print_score(filename):
    with open(filename) as f:
        text = f.read()
        p = perplexity.perplexity(text)
        f = finewebed.finewebed(text)
        print(f"{filename}: Perplexity: {p} | Quality: {perplexity.category(p)}")
        print(f"{filename}: FineWebEd:  {f} | {finewebed.category(f)}")
        print("")

executor = ThreadPoolExecutor(20).map(print_score, DATA_FILES)
