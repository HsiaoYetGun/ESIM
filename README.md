# Notice

There are some problems with this version code (the mask of attention weight [Model.py, line 160-170] and the mask of mean and max [Model.py, line 220-225]), please don't use this code directly!

I’m too busy recently to follow this repo, and I will update this code in my winter vacation (starting from the 26th, Jan).

# ESIM

A Tensorflow implementation of Chen-Qian's **[Enhanced LSTM for Natural Language Inference](https://arxiv.org/abs/1609.06038)** from ACL 2017.

# Dataset

The dataset used for this task is [Stanford Natural Language Inference (SNLI)](https://nlp.stanford.edu/projects/snli/). Pretrained [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) obtained from common crawl with 840B tokens used for words.

# Requirements

* Python>=3
* NumPy
* TensorFlow>=1.8

# Usage

Download dataset from [Stanford Natural Language Inference](https://nlp.stanford.edu/projects/snli/), then move `snli_1.0_train.jsonl`, `snli_1.0_dev.jsonl`, `snli_1.0_test.jsonl` into `./SNLI/raw data`.

```com
# move dataset to the right place
mkdir -p ./SNLI/raw\ data
mv snli_1.0_*.jsonl ./SNLI/raw\ data
```

Data preprocessing for convert source data into an easy-to-use format.

```python
python3 Utils.py
```

Default hyper-parameters have been stored in config file in the path of `./config/config.yaml`.

Training model:

```python
python3 Train.py
```

Test model:

```python
python3 Test.py
```