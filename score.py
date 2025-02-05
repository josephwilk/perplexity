import sys

if(len(sys.argv) < 2):
    print("Usage: python score.py FILENAME")
    exit(1)
    
target = sys.argv[1]

import finewebed
import perplexity

with open(target) as f:
    text = f.read()
    p = perplexity.perplexity(text)
    f = finewebed.finewebed(text)

    print(f"{target}: Perplexity: {p} | Quality: {perplexity.category(p)}")
    print(f"{target}: FineWebEd:  {f} | {finewebed.category(f)}")


