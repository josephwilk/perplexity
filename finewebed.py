import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")
model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")

def finewebed(text):
    inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits.squeeze(-1).float().detach().numpy()
    score = logits.item()
    return score

def category(score):
    #int_score >= 3 as a threshold for data curation
    if score >= 3:
        return "Accept"
    else:
        return "Reject"

if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print("Usage: python perplexity FILENAME")
        exit(1)
    
    target = sys.argv[1]

    with open(target) as f:
        text = f.read()
        s = finewebed(text)
        print(s)
