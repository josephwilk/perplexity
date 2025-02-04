import kenlm
import pandas as pd
import sentencepiece

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
        #n = text_normalizer.normalize(line)
        #clean_content = sp.encode_as_pieces(n)
        log_score = model.score(line)
        length = len(line.split()) + 1
        doc_log_score += log_score
        doc_length += length
        sentences.append(f"{pp(log_score, length)}\t{line}")

    doc_score = round(pp(doc_log_score, doc_length), 1)
        
    print("\n".join(sentences))
    print(f"doc score: {doc_score}")
    print(cutoff_bucket(cutoffs, doc_score))
    print("\n")
    return doc_score

def cutoff_bucket(cutoffs, perplexity):
    lang = "en"
    if perplexity < 0:
        return "all"
    pp_head, pp_tail = cutoffs[lang]
    if perplexity < pp_head:
        return "head"
    if perplexity < pp_tail:
        return "middle"
    return "tail"


random_content = '''Causus
could assist a friend. A short time later, the tea parties will influence policy at all? Hume's injunction underlies the caution of scientists from the article: “Invariably,” says Craig, “a black-themed book will come to think about gay/straight alliances on Catholic campuses? Do they subtract the max from each town work, shop, eat, and socialize in towns separate from local school districts build new schools. I think plenty of opportunities for new tevee conference</p> <p>Heading to Frogtown for the sake of characters, my list here can’t even '''

wiki_content = '''
Prostate cancer is the uncontrolled growth of cells in the prostate, a gland in the male reproductive system below the bladder. Abnormal growth of prostate tissue is usually detected through screening tests, typically blood tests that check for prostate-specific antigen (PSA) levels. Those with high levels of PSA in their blood are at increased risk for developing prostate cancer. Diagnosis requires a biopsy of the prostate. If cancer is present, the pathologist assigns a Gleason score, and a higher score represents a more dangerous tumor. Medical imaging is performed to look for cancer that has spread outside the prostate. Based on the Gleason score, PSA levels, and imaging results, a cancer case is assigned a stage 1 to 4. A higher stage signifies a more advanced, more dangerous disease.

Most prostate tumors remain small and cause no health problems. These are managed with active surveillance, monitoring the tumor with regular tests to ensure it has not grown. Tumors more likely to be dangerous can be destroyed with radiation therapy or surgically removed by radical prostatectomy. Those whose cancer spreads beyond the prostate are treated with hormone therapy which reduces levels of the androgens (male sex hormones) that prostate cells need to survive. Eventually cancer cells can grow resistant to this treatment. This most-advanced stage of the disease, called castration-resistant prostate cancer, is treated with continued hormone therapy alongside the chemotherapy drug docetaxel. Some tumors metastasize (spread) to other areas of the body, particularly the bones and lymph nodes. There, tumors cause severe bone pain, leg weakness or paralysis, and eventually death. Prostate cancer prognosis depends on how far the cancer has spread at diagnosis. Most men diagnosed have tumors confined to the prostate; 99% of them survive more than 10 years from their diagnoses. Tumors that have metastasized to distant body sites are most dangerous, with five-year survival rates of 30–40%.

The risk of developing prostate cancer increases with age; the average age of diagnosis is 67. Those with a family history of any cancer are more likely to have prostate cancer, particularly those who inherit cancer-associated variants of the BRCA2 gene. Each year 1.2 million cases of prostate cancer are diagnosed, and 350,000 die of the disease,[2] making it the second-leading cause of cancer and cancer death in men. One in eight men is diagnosed with prostate cancer in his lifetime and one in forty dies of the disease.[3] Prostate tumors were first described in the mid-19th century, during surgeries on men with urinary obstructions. Initially, prostatectomy was the primary treatment for prostate cancer. By the mid-20th century, radiation treatments and hormone therapies were developed to improve prostate cancer treatment. The invention of hormone therapies for prostate cancer was recognized with the 1966 Nobel Prize to Charles B. Huggins and the 1977 Prize to Andrzej W. Schally.
'''

perplexity("A A A A A")
perplexity("<b>A</b> A A A A")
perplexity(wiki_content)
perplexity(random_content)
