import sys
import glob
from concurrent.futures import ThreadPoolExecutor

import finewebed
import perplexity

if(len(sys.argv) < 2):
    print("Usage: python score.py DIRNAME")
    exit(1)
    
target_dir = sys.argv[1]
DATA_FILES = glob.glob(target_dir+'**/*.txt', recursive=True)

def print_score(filename):
    try:
        with open(filename, 'r') as f:
            text = f.read()
            p = perplexity.perplexity(text)
            f = finewebed.finewebed(text)
            print(f"{filename}: Perplexity: {p} | Quality: {perplexity.category(p)}")
            print(f"{filename}: FineWebEd:  {f} | {finewebed.category(f)}")
            print("")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

with ThreadPoolExecutor(max_workers=20) as executor:
    executor.map(print_score, DATA_FILES)
