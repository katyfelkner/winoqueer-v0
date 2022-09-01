# winoqueer
Benchmark dataset for anti-queer bias in large language models (LLMs)

Our paper, [Towards Winoqueer: Developing a Benchmark for Anti-Queer Bias in Large Language Models](https://arxiv.org/abs/2206.11484),  was published in the QueerInAI workshop at NAACL 2022!

## Repo contents:
### Finetuning Data
Our news and twitter datasets are included as .pkl files. Unpickle them to view and use our data!

### Finetuning Scripts
Scripts use to preprocess data (segment and normalize) and finetune models. Tweets are normalized using [TweetNormalizer from BERTweet](https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py).

### Model Checkpoints
Model Checkpoints are included for four models (BERT_base, BERT_large, SpanBERT_base, SpanBERT_large) under three finetuning conditions (none, LGBTQ+ news, LGBTQ+ twitter).


### Benchmark Data
`winoqueer_benchmark.csv` is the benchmark data used in our experiments in the paper. Use this to replicate our results!

Our data follows the [CrowS-Pairs](https://github.com/nyu-mll/crows-pairs) format, and you should use their evaluation script to run our metric. 

### Note
Some files in this repo are large. You will probably need to use Git LFS.