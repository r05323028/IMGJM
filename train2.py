# Copyright (c) 2020 seanchang
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import Dict, Tuple
from argparse import ArgumentParser
import coloredlogs
import logging
import yaml
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm, trange
from IMGJM import IMGJM2
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
    arg_parser.add_argument('--learning_rate', default=0.001, type=float)
    arg_parser.add_argument('--batch_size', type=int, default=32)
    arg_parser.add_argument('--epochs', type=int, default=50)
    arg_parser.add_argument('--model_dir', type=str, default='outputs')
    arg_parser.add_argument('--model_config_fp',
                            type=str,
                            default='model_settings.yml')
    arg_parser.add_argument('--embedding', type=str, default='glove')
    arg_parser.add_argument('--dataset', type=str, default='laptop')
    return vars(arg_parser.parse_args())


def load_model_config(file_path: str) -> Dict:
    with open(file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        return config


def run_optimization(inputs, optimizer, model):
    with tf.GradientTape() as g:
        multi_grained_target, multi_grained_sentiment = model(inputs,
                                                              training=True)
        total_loss = model.get_total_loss(inputs)
    trainable_variables = model.trainable_variables
    gradients = g.gradient(total_loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))


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
    optimizer = tf.optimizers.Adam(kwargs.get('learning_rate'))
    config = load_model_config(kwargs.get('model_config_fp'))
    if kwargs.get('embedding') == 'fasttext':
        model = IMGJM2(char_vocab_size=vocab_size,
                       embedding_size=dataset.fasttext_model.get_dimension(),
                       input_type='embedding',
                       dropout=False,
                       **config['custom'])
    else:
        model = IMGJM2(char_vocab_size=vocab_size,
                       embedding_matrix=embedding_weights,
                       input_type='ids',
                       dropout=False,
                       **config['custom'])
    if kwargs.get('embedding') == 'fasttext':
        tf_data = tf.data.Dataset.from_tensor_slices(
            dataset.pad_train_data).batch(32)
    else:
        tf_data = tf.data.Dataset.from_tensor_slices(
            dataset.pad_train_data).batch(32)
    target_f1_score = tfa.metrics.F1Score(num_classes=int(model.C_tar),
                                          dtype=tf.int32)
    for inputs in tf_data.take(100):
        run_optimization(inputs, optimizer, model)
        target_preds, sentiment_preds = model.crf_decode(inputs)
        print(target_preds)


if __name__ == '__main__':
    kwargs = get_args()
    main(**kwargs)