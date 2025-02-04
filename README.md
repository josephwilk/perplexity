Very simple example of a Perplexity calculation based on wikipedia trained model.
Based on filtering the common crawl from: https://github.com/facebookresearch/cc_net & accompanying paper: https://arxiv.org/pdf/1911.00359



# Install

```
pip install -r dependencies.txt
```

# Fetch models

```
make
```


# Run

```
python perplexity.py examples/wikipedia.txt
python perplexity.py examples/markov.txt
```
