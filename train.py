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
    arg_parser.add_argument('--learning_rate', default=0.001, type=float)
    arg_parser.add_argument('--batch_size', type=int, default=32)
    arg_parser.add_argument('--epochs', type=int, default=50)
    arg_parser.add_argument('--model_dir', type=str, default='outputs')
    arg_parser.add_argument('--model_config_fp',
                            type=str,
                            default='model_settings.yml')
    arg_parser.add_argument('--embedding', type=str, default='glove')
    arg_parser.add_argument('--dataset', type=str, default='laptop')
    arg_parser.add_argument('--checkpoint_step', type=int, default=100)
    arg_parser.add_argument('--logdir', type=str, default='logs')
    return vars(arg_parser.parse_args())


def load_model_config(file_path: str) -> Dict:
    with open(file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        return config


def run_optimization(inputs: Tuple, optimizer: tf.optimizers.Optimizer,
                     model: tf.keras.Model, mask: tf.Tensor) -> tf.Tensor:
    with tf.GradientTape() as g:
        total_loss = model.get_total_loss(inputs, mask=mask, training=True)
    trainable_variables = model.trainable_variables
    gradients = g.gradient(total_loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return total_loss


def test(model: tf.keras.Model,
         data: tf.data.Dataset,
         max_seq_len: int,
         average: str = 'micro'):
    test_target_f1 = tfa.metrics.F1Score(num_classes=model.C_tar,
                                         average=average)
    test_sentiment_f1 = tfa.metrics.F1Score(num_classes=model.C_sent,
                                            average=average)
    test_pbar = tqdm(data, desc='testing')
    for i, inputs in enumerate(test_pbar):
        mask = tf.sequence_mask(inputs[2], max_seq_len)
        target_pred, sent_pred = model.crf_decode(inputs, mask=mask)
        test_target_f1(y_true=inputs[3],
                       y_pred=tf.cast(target_pred, tf.float32))
        test_target_precision = test_target_f1.true_positives / (
            test_target_f1.false_positives + test_target_f1.true_positives)
        test_target_recall = test_target_f1.true_positives / (
            test_target_f1.false_negatives + test_target_f1.true_positives)
        test_sentiment_f1(y_true=inputs[4],
                          y_pred=tf.cast(sent_pred, tf.float32))
        test_sentiment_precision = test_sentiment_f1.true_positives / (
            test_sentiment_f1.false_positives +
            test_sentiment_f1.true_positives)
        test_sentiment_recall = test_sentiment_f1.true_positives / (
            test_sentiment_f1.false_negatives +
            test_sentiment_f1.true_positives)
        test_pbar.set_description(
            f'[Test][Target][Average: {average}]P: {test_target_precision.numpy():.3f} R: {test_target_recall.numpy():.3f} F1: {test_target_f1.result().numpy():.3f} [Senti]P: {test_sentiment_precision.numpy():.3f} R: {test_sentiment_recall.numpy():.3f} F1: {test_sentiment_f1.result().numpy():.3f}'
        )


def train(model: tf.keras.Model,
          train_data: tf.data.Dataset,
          train_max_seq_len: int,
          optimizer: tf.optimizers.Optimizer,
          test_data: tf.data.Dataset = None,
          test_max_seq_len: int = None,
          checkpoint_step: int = 100,
          average: str = 'micro',
          logdir: str = 'logs'):
    metrics_file_writer = tf.summary.create_file_writer(logdir + '/metrics')
    metrics_file_writer.set_as_default()
    target_f1 = tfa.metrics.F1Score(num_classes=model.C_tar, average=average)
    sentiment_f1 = tfa.metrics.F1Score(num_classes=model.C_sent,
                                       average=average)
    train_pbar = tqdm(train_data, desc='training')
    for i, inputs in enumerate(train_pbar):
        train_mask = tf.sequence_mask(inputs[2], train_max_seq_len)
        total_loss = run_optimization(inputs,
                                      optimizer,
                                      model,
                                      mask=train_mask)
        target_pred, sent_pred = model.crf_decode(inputs, mask=train_mask)
        target_f1(y_true=inputs[3], y_pred=tf.cast(target_pred, tf.float32))
        target_precision = target_f1.true_positives / (
            target_f1.false_positives + target_f1.true_positives)
        target_recall = target_f1.true_positives / (target_f1.false_negatives +
                                                    target_f1.true_positives)
        sentiment_f1(y_true=inputs[4], y_pred=tf.cast(sent_pred, tf.float32))
        sentiment_precision = sentiment_f1.true_positives / (
            sentiment_f1.false_positives + sentiment_f1.true_positives)
        sentiment_recall = sentiment_f1.true_positives / (
            sentiment_f1.false_negatives + sentiment_f1.true_positives)
        train_pbar.set_description(
            f'[Train][Target][Average: {average}]P: {target_precision.numpy():.3f} R: {target_recall.numpy():.3f} F1: {target_f1.result().numpy():.3f} [Senti]P: {sentiment_precision.numpy():.3f} R: {sentiment_recall.numpy():.3f} F1: {sentiment_f1.result().numpy():.3f}'
        )
        tf.summary.scalar('total loss',
                          data=tf.reduce_mean(total_loss),
                          step=i)
        tf.summary.scalar('train target precision',
                          data=target_precision.numpy(),
                          step=i)
        tf.summary.scalar('train target recall',
                          data=target_recall.numpy(),
                          step=i)
        tf.summary.scalar('train target f1',
                          data=target_f1.result().numpy(),
                          step=i)
        tf.summary.scalar('train sentiment precision',
                          data=sentiment_precision.numpy(),
                          step=i)
        tf.summary.scalar('train sentiment recall',
                          data=sentiment_recall.numpy(),
                          step=i)
        tf.summary.scalar('train sentiment f1',
                          data=sentiment_f1.result().numpy(),
                          step=i)
        if i % checkpoint_step == 0 and i != 0:
            if test_data:
                test(model, data=test_data, max_seq_len=test_max_seq_len)
            model.save_weights('outputs/IMGJM', save_format='tf')


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
    config = load_model_config(kwargs.get('model_config_fp'))
    optimizer = tf.optimizers.Adam(config['custom'].get('learning_rate'))
    if kwargs.get('embedding') == 'fasttext':
        model = IMGJM(char_vocab_size=vocab_size,
                      embedding_size=dataset.fasttext_model.get_dimension(),
                      input_type='embedding',
                      dropout=False,
                      **config['custom'])
    else:
        model = IMGJM(char_vocab_size=vocab_size,
                      embedding_matrix=embedding_weights,
                      input_type='ids',
                      dropout=False,
                      **config['custom'])
    if kwargs.get('embedding') == 'fasttext':
        temp_train_data = dataset.pad_train_data
        tf_train_data = tf.data.Dataset.from_tensor_slices(
            temp_train_data).repeat(kwargs.get('epochs')).batch(
                kwargs.get('batch_size'))
        tf_test_data = tf.data.Dataset.from_tensor_slices(
            dataset.pad_test_data).repeat(1).batch(kwargs.get('batch_size'))
    else:
        temp_train_data = dataset.pad_train_data
        tf_train_data = tf.data.Dataset.from_tensor_slices(
            temp_train_data).repeat(kwargs.get('epochs')).batch(
                kwargs.get('batch_size'))
        tf_test_data = tf.data.Dataset.from_tensor_slices(
            dataset.pad_test_data).repeat(1).batch(kwargs.get('batch_size'))
    train(model=model,
          train_data=tf_train_data,
          train_max_seq_len=dataset.train_max_seq_len,
          optimizer=optimizer,
          test_data=tf_test_data,
          test_max_seq_len=dataset.test_max_seq_len,
          checkpoint_step=kwargs.get('checkpoint_step'))


if __name__ == '__main__':
    kwargs = get_args()
    main(**kwargs)