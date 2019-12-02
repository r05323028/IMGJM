# Copyright (c) 2019 seanchang
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
'''
Demo script for IMGJM
'''
from typing import Dict, Tuple, List
from argparse import ArgumentParser
import logging
import yaml
import coloredlogs
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from IMGJM import IMGJM
from IMGJM.data import (SemEval2014, Twitter, KKBOXSentimentData)
from IMGJM.utils import build_glove_embedding, build_mock_embedding


class BoolParser:
    @classmethod
    def parse(cls, arg: str) -> bool:
        if arg.lower() in ['false', 'no']:
            return False
        else:
            return True


def get_logger(logger_name: str = 'IMGJM',
               level: str = 'INFO') -> logging.Logger:
    logger = logging.getLogger(logger_name)
    coloredlogs.install(
        level=level,
        fmt=
        f'%(asctime)s | %(name)-{len(logger_name) + 1}s| %(levelname)s | %(message)s',
        logger=logger)
    return logger


def get_args() -> Dict:
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--model_dir', type=str, default='outputs')
    arg_parser.add_argument('--model_config_fp',
                            type=str,
                            default='model_settings.yml')
    arg_parser.add_argument('--embedding', type=str, default='glove')
    arg_parser.add_argument('--dataset', type=str, default='laptop')
    return vars(arg_parser.parse_args())


def get_sentiment_clue_vis(word: List, target: List, sentiment: List,
                           sentiment_clue: List):
    fig, axes = plt.subplots(nrows=3, ncols=1)
    axes[0].matshow(sentiment_clue, cmap='Blues')
    axes[0].set_xticklabels([''] + word[0])
    axes[0].yaxis.set_visible(False)
    axes[0].xaxis.set_major_locator(MultipleLocator(1))
    axes[0].xaxis.set_ticks_position('bottom')
    axes[0].set_title('Sentiment Clue')
    axes[1].matshow(target, cmap='Blues')
    axes[1].set_xticklabels([''] + word[0])
    axes[1].yaxis.set_visible(False)
    axes[1].xaxis.set_major_locator(MultipleLocator(1))
    axes[1].xaxis.set_ticks_position('bottom')
    for i, tar in enumerate(target[0]):
        if tar == 1:
            axes[1].text(i, 0, s='B-PER', ha='center', va='center')
        elif tar == 2:
            axes[1].text(i, 0, s='I-PER', ha='center', va='center')
        else:
            axes[1].text(i, 0, s='O', ha='center', va='center')
    axes[1].set_title('Target Prediction')
    axes[2].matshow(sentiment, cmap='Blues')
    axes[2].set_xticklabels([''] + word[0])
    axes[2].yaxis.set_visible(False)
    axes[2].xaxis.set_major_locator(MultipleLocator(1))
    axes[2].xaxis.set_ticks_position('bottom')
    for i, sent in enumerate(sentiment[0]):
        if sent == 1:
            axes[2].text(i, 0, s='B-POS', ha='center', va='center')
        elif sent == 2:
            axes[2].text(i, 0, s='I-POS', ha='center', va='center')
        elif sent == 3:
            axes[2].text(i, 0, s='B-NEU', ha='center', va='center')
        elif sent == 4:
            axes[2].text(i, 0, s='I-NEU', ha='center', va='center')
        elif sent == 5:
            axes[2].text(i, 0, s='B-NEG', ha='center', va='center')
        elif sent == 6:
            axes[2].text(i, 0, s='I-NEG', ha='center', va='center')
        else:
            axes[2].text(i, 0, s='O', ha='center', va='center')
    axes[2].set_title('Sentiment Prediction')
    fig.suptitle('IMGJM Prediction Visualization')
    plt.show()


def build_feed_dict(input_tuple: Tuple[np.ndarray],
                    input_type: str = 'ids') -> Dict:
    if input_type == 'ids':
        pad_char_ids, pad_word_ids, sequence_length, pad_entities, pad_polarities = input_tuple
        feed_dict = {
            'char_ids': pad_char_ids,
            'word_ids': pad_word_ids,
            'sequence_length': sequence_length,
            'y_target': pad_entities,
            'y_sentiment': pad_polarities,
        }
        return feed_dict
    else:
        pad_char_ids, pad_word_embedding, sequence_length, pad_entities, pad_polarities = input_tuple
        feed_dict = {
            'char_ids': pad_char_ids,
            'word_embedding': pad_word_embedding,
            'sequence_length': sequence_length,
            'y_target': pad_entities,
            'y_sentiment': pad_polarities,
        }
        return feed_dict


def load_model_config(file_path: str) -> Dict:
    with open(file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        return config


def main(*args, **kwargs):
    np.random.seed(1234)
    logger = get_logger()
    if kwargs.get('embedding') == 'mock':
        logger.info('Initializing dataset...')
        if kwargs.get('dataset') == 'laptop':
            dataset = SemEval2014(resource='laptop')
        elif kwargs.get('dataset') == 'rest':
            dataset = SemEval2014(resource='rest')
        elif kwargs.get('dataset') == 'kkbox':
            dataset = KKBOXSentimentData()
        else:
            dataset = Twitter()
        vocab_size = len(dataset.char2id)
        logger.info('Dataset loaded.')
        logger.info('Build mock embedding')
        _, embedding_weights = build_mock_embedding(dataset.word2id)
        logger.info('Building mock embedding finished')
    elif kwargs.get('embedding') == 'glove':
        logger.info('Loading Glove embedding...')
        word2id, embedding_weights, _ = build_glove_embedding()
        logger.info('Embeding loaded.')
        logger.info('Initializing dataset...')
        if kwargs.get('dataset') == 'laptop':
            dataset = SemEval2014(word2id=word2id, resource='laptop')
        elif kwargs.get('dataset') == 'rest':
            dataset = SemEval2014(word2id=word2id, resource='rest')
        elif kwargs.get('dataset') == 'kkbox':
            dataset = KKBOXSentimentData(word2id=word2id)
        else:
            dataset = Twitter(word2id=word2id)
        vocab_size = len(dataset.char2id)
        logger.info('Dataset loaded.')
    elif kwargs.get('embedding') == 'fasttext':
        logger.info('Initializing dataset...')
        if kwargs.get('dataset') == 'laptop':
            dataset = SemEval2014(resource='laptop')
        elif kwargs.get('dataset') == 'rest':
            dataset = SemEval2014(resource='rest')
        elif kwargs.get('dataset') == 'kkbox':
            dataset = KKBOXSentimentData()
        else:
            dataset = Twitter()
        vocab_size = len(dataset.char2id)
        logger.info('Dataset loaded.')
    else:
        logger.warning('Invalid embedding choice.')
    logger.info('Loading model...')
    config = load_model_config(kwargs.get('model_config_fp'))
    if kwargs.get('embedding') == 'fasttext':
        model = IMGJM(char_vocab_size=vocab_size,
                      embedding_size=dataset.fasttext_model.get_dimension(),
                      input_type='embedding',
                      dropout=False,
                      deploy=True,
                      **config['custom'])
    else:
        model = IMGJM(char_vocab_size=vocab_size,
                      embedding_weights=embedding_weights,
                      dropout=False,
                      deploy=True,
                      **config['custom'])
    logger.info('Model loaded.')
    model.load_model('outputs' + '/' + 'model')
    s = [
        "i ordered my 2012 mac mini after being disappointed with spec of the new 27 ' imacs ."
    ]
    inputs = dataset.merge_and_pad_all(s)
    if kwargs.get('embedding') == 'fasttext':
        feed_dict = build_feed_dict(inputs, input_type='embedding')
    else:
        feed_dict = build_feed_dict(inputs, input_type='ids')
    target_preds, sentiment_preds = model.predict_on_batch(feed_dict)
    get_sentiment_clue_vis([sent.split(' ') for sent in s], target_preds,
                           sentiment_preds,
                           model.get_sentiment_clue(feed_dict)[:, :, 1])


if __name__ == '__main__':
    kwargs = get_args()
    main(**kwargs)