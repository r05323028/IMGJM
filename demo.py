'''
Demo script for IMGJM
'''
from typing import Dict, Tuple, List
from argparse import ArgumentParser
import logging
import coloredlogs
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from IMGJM import IMGJM
from IMGJM.data import (SemEval2014, Twitter)
from IMGJM.utils import build_glove_embedding, build_mock_embedding


class BoolParser:
    @classmethod
    def parse(cls, arg: str) -> bool:
        if arg.lower() in ['false', 'no']:
            return False
        else:
            return True


def get_logger(logger_name: str = 'IMGJM', level: str = 'INFO'):
    logger = logging.getLogger(logger_name)
    coloredlogs.install(
        level=level,
        fmt=
        f'%(asctime)s | %(name)-{len(logger_name) + 1}s| %(levelname)s | %(message)s',
        logger=logger)
    return logger


def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--batch_size', type=int, default=32)
    arg_parser.add_argument('--epochs', type=int, default=3)
    arg_parser.add_argument('--model_dir', type=str, default='outputs')
    arg_parser.add_argument('--mock_embedding',
                            type=BoolParser.parse,
                            default=False)
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
                    embedding_weights: np.ndarray) -> Dict:
    pad_char_ids, pad_word_ids, sequence_length, pad_entities, pad_polarities = input_tuple
    feed_dict = {
        'char_ids': pad_char_ids,
        'word_ids': pad_word_ids,
        'sequence_length': sequence_length,
        'y_target': pad_entities,
        'y_sentiment': pad_polarities,
        'glove_embedding': embedding_weights,
    }
    return feed_dict


def main(*args, **kwargs):
    np.random.seed(1234)
    logger = get_logger()
    if kwargs.get('mock_embedding'):
        logger.info('Initializing dataset...')
        dataset = SemEval2014()
        vocab_size = len(dataset.char2id)
        logger.info('Dataset loaded.')
        logger.info('Build mock embedding')
        _, embedding_weights = build_mock_embedding(dataset.word2id)
        logger.info('Building mock embedding finished')
    else:
        logger.info('Loading Glove embedding...')
        word2id, embedding_weights, _ = build_glove_embedding()
        logger.info('Embeding loaded.')
        logger.info('Initializing dataset...')
        dataset = SemEval2014(word2id=word2id)
        vocab_size = len(dataset.char2id)
        logger.info('Dataset loaded.')
    model = IMGJM(char_vocab_size=vocab_size,
                  embedding_weights=embedding_weights,
                  batch_size=kwargs.get('batch_size'),
                  deploy=True)
    s = ["best spicy/p tuna roll , great asian salad ."]
    model.load_model('outputs' + '/' + 'model')
    inputs = dataset.merge_and_pad_all(s)
    feed_dict = build_feed_dict(inputs, embedding_weights)
    target_preds, sentiment_preds = model.predict_on_batch(feed_dict)
    get_sentiment_clue_vis([sent.split(' ') for sent in s], target_preds,
                           sentiment_preds,
                           model.get_sentiment_clue(feed_dict)[:, :, 1])


if __name__ == '__main__':
    kwargs = get_args()
    main(**kwargs)