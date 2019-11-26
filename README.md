# Interactive Multi-Grained Joint Model for Targeted Sentiment Analysis

This repo is the TensorFlow implementation of **Interactive Multi-Grained Joint Model for Targeted Sentiment Analysis** which was accepted by ACM CIKM 2019.

![Model](model.png)

## Usage

WIP

## Performance

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
| SemEval2014-Laptop     | GloVe     | 0.676     | 0.327  | 0.441 |
| SemEval2014-Restaurant | GloVe     | 0.587     | 0.265  | 0.366 |
| Twitter                | GloVe     | 0.0       | 0.0    | 0.0   |

#### Test

| Dataset                | Embedding | Precision | Recall | F1    |
| ---------------------- | --------- | --------- | ------ | ----- |
| SemEval2014-Laptop     | GloVe     | 0.823     | 0.284  | 0.423 |
| SemEval2014-Restaurant | GloVe     | 0.877     | 0.188  | 0.310 |
| Twitter                | GloVe     | 0.0       | 0.0    | 0.0   |

### Sentiment Analysis

#### Train

| Dataset                | Precision | Recall | F1    |
| ---------------------- | --------- | ------ | ----- |
| SemEval2014-Laptop     | 0.618     | 0.256  | 0.362 |
| SemEval2014-Restaurant | 0.257     | 0.217  | 0.236 |
| Twitter                | 0.0       | 0.0    | 0.0   |

#### Test

| Dataset                | Precision | Recall | F1    |
| ---------------------- | --------- | ------ | ----- |
| SemEval2014-Laptop     | 0.595     | 0.159  | 0.251 |
| SemEval2014-Restaurant | 0.727     | 0.135  | 0.228 |
| Twitter                | 0.0       | 0.0    | 0.0   |

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
