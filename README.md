# Interactive Multi-Grained Joint Model for Targeted Sentiment Analysis

This repository is the TensorFlow implementation of **Interactive Multi-Grained Joint Model for Targeted Sentiment Analysis** which was accepted by ACM CIKM 2019.

![Model](model.png)

## Usage

### Environment Setup

Just use pipenv to install requirements.

```bash
pipenv install
```

### Train Model

```bash
pipenv run python train.py --mock_embedding=[True|False] --epochs=[nums]
```

## Performance

- Word embedding: GloVe.840B.300D
- Model setting
  - epochs: 50
  - learning_rate: 0.001
  - dropout_rate: 0.5
  - batch_size: 32
  - kernel_size: 3
  - filter_nums: 50
  - C_tar: 3
  - C_sent: 7
  - beta: 1.0
  - gamma: 0.7

### Entity Extraction

#### Train

| Dataset                | Embedding | Precision | Recall | F1    |
| ---------------------- | --------- | --------- | ------ | ----- |
| SemEval2014-Laptop     | GloVe     | 0.416     | 0.353  | 0.382 |
| SemEval2014-Restaurant | GloVe     | 0.463     | 0.267  | 0.339 |
| Twitter                | GloVe     | 0.834     | 0.886  | 0.859 |

#### Test

| Dataset                | Embedding | Precision | Recall | F1    |
| ---------------------- | --------- | --------- | ------ | ----- |
| SemEval2014-Laptop     | GloVe     | 0.791     | 0.329  | 0.465 |
| SemEval2014-Restaurant | GloVe     | 0.884     | 0.189  | 0.311 |
| Twitter                | GloVe     | 0.992     | 0.990  | 0.991 |

### Sentiment Analysis

#### Train

| Dataset                | Embedding | Precision | Recall | F1    |
| ---------------------- | --------- | --------- | ------ | ----- |
| SemEval2014-Laptop     | GloVe     | 0.574     | 0.261  | 0.359 |
| SemEval2014-Restaurant | GloVe     | 0.583     | 0.216  | 0.315 |
| Twitter                | GloVe     | 0.731     | 0.682  | 0.706 |

#### Test

| Dataset                | Embedding | Precision | Recall | F1    |
| ---------------------- | --------- | --------- | ------ | ----- |
| SemEval2014-Laptop     | GloVe     | 0.612     | 0.169  | 0.265 |
| SemEval2014-Restaurant | GloVe     | 0.778     | 0.123  | 0.213 |
| Twitter                | GloVe     | 0.681     | 0.676  | 0.678 |

## Citation

You can cite this [paper](https://dl.acm.org/citation.cfm?id=3357384.3358024) if you use this model

```
@inproceedings{Yin:2019:IMJ:3357384.3358024,
 author = {Yin, Da and Liu, Xiao and Wan, Xiaojun},
 title = {Interactive Multi-Grained Joint Model for Targeted Sentiment Analysis},
 booktitle = {Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
 series = {CIKM '19},
 year = {2019},
 isbn = {978-1-4503-6976-3},
 location = {Beijing, China},
 pages = {1031--1040},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3357384.3358024},
 doi = {10.1145/3357384.3358024},
 acmid = {3358024},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {interaction mechanism, joint model, multi-grained model, neural networks, sentiment analysis, sequence labeling},
} 
```

## License

[MIT](LICENSE)
