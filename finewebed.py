import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification


if(len(sys.argv) < 2):
    print("Usage: python perplexity FILENAME")
    exit(1)
    
target = sys.argv[1]



tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")
model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")


with open(target) as f:
    text = f.read()
    inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits.squeeze(-1).float().detach().numpy()
    score = logits.item()
    result = {
        "text": text,
        "score": score,
        "int_score": int(round(max(0, min(score, 5)))),
    }
   
    print(result)
