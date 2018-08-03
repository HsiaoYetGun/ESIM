# ESIM
A Tensorflow implementation of Chen-Qian's **[Enhanced LSTM for Natural Language Inference](https://arxiv.org/abs/1609.06038)** from ACL 2016.

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