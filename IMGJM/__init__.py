# Copyright (c) 2019 seanchang
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from abc import ABCMeta
from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tf_metrics
# from IMGJM.layers import (CharEmbedding, GloveEmbedding, CoarseGrainedLayer,
#                           Interaction, FineGrainedLayer)
from IMGJM.layers import (CharEmbedding, GloveEmbedding, CoarseGrainedLayer,
                          Interaction, FineGrainedLayer)


class IMGJM(tf.keras.Model):
    '''
    Interactive Multi-Grained Joint Model for Targeted Sentiment Analysis (CIKM 2019)

    Attributes:
        char_vocab_size (int)
        embedding_weights (np.ndarray)
        batch_size (int)
        learning_rate (float)
        embedding_size (int)
        hidden_nums (int)
        dropout (bool)
        dropout_rate (float)
        kernel_size (int)
        filter_nums (int)
        C_tar (int)
        C_sent (int)
        beta (float)
        gamma (float)
    '''
    def __init__(self,
                 char_vocab_size: int,
                 embedding_matrix: np.ndarray = None,
                 embedding_size: int = 300,
                 input_type: str = 'ids',
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 hidden_nums: int = 700,
                 dropout: bool = True,
                 dropout_rate: float = 0.5,
                 kernel_size: int = 3,
                 filter_nums: int = 50,
                 C_tar: int = 5,
                 C_sent: int = 7,
                 beta: float = 0.7,
                 char_mask_value: int = 0,
                 word_mask_value: int = 0,
                 *args,
                 **kwargs):
        super(IMGJM, self).__init__()
        self.char_vocab_size = char_vocab_size
        self.input_type = input_type
        if input_type == 'ids':
            self.embedding_matrix = embedding_matrix
            self.embedding_shape = embedding_matrix.shape
        else:
            self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.char_mask_value = char_mask_value
        self.word_mask_value = word_mask_value
        self.filter_nums = filter_nums
        self.hidden_nums = hidden_nums
        self.C_tar = C_tar
        self.C_sent = C_sent

        # layers initialization
        self.dropout_layer = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.char_mask_layer = tf.keras.layers.Masking(
            mask_value=self.char_mask_value)
        self.word_mask_layer = tf.keras.layers.Masking(
            mask_value=self.word_mask_value)
        self.char_embedding_layer = CharEmbedding(vocab_size=char_vocab_size)
        if self.input_type == 'ids':
            self.word_embedding_layer = GloveEmbedding(self.embedding_matrix)
        self.coarse_grained_layer = CoarseGrainedLayer(
            hidden_nums=self.hidden_nums)
        self.interaction_layer = Interaction(hidden_nums=self.hidden_nums,
                                             C_tar=self.C_tar,
                                             C_sent=self.C_sent)
        self.fine_grained_layer = FineGrainedLayer(
            hidden_nums=self.hidden_nums, C_tar=self.C_tar, C_sent=self.C_sent)

        # params initialization
        self.target_trans_params = np.zeros(shape=(self.C_tar, self.C_tar))
        self.sentiment_trans_params = np.zeros(shape=(self.C_sent, C_sent))

    def get_loss(self, multi_grained: tf.Tensor, labels: tf.Tensor,
                 sequence_lengths: tf.Tensor) -> tf.Tensor:
        log_likelihood, trans_params = tfa.text.crf_log_likelihood(
            inputs=multi_grained,
            tag_indices=labels,
            sequence_lengths=sequence_lengths)
        return tf.reduce_mean(-log_likelihood), trans_params

    def get_loss_ol(self, coarse_grained_target: tf.Tensor,
                    sentiment_clue: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(coarse_grained_target[:, :, 0] *
                              sentiment_clue[:, :, 0])

    def get_loss_brl(self, multi_grained_target: tf.Tensor,
                     multi_grained_sentiment: tf.Tensor) -> tf.Tensor:
        return tf.losses.mean_squared_error(
            1 - tf.nn.softmax(multi_grained_target)[:, :, 0],
            1 - tf.nn.softmax(multi_grained_sentiment)[:, :, 0])

    def get_total_loss(self,
                       inputs: Tuple[tf.Tensor],
                       beta: float = 1.0,
                       gamma: float = 0.7) -> tf.Tensor:
        _, _, seq_len, y_target, y_sent = inputs
        target_loss, self.target_trans_params = self.get_loss(
            self.multi_grained_target, y_target, seq_len)
        sent_loss, self.sentiment_trans_params = self.get_loss(
            self.multi_grained_sentiment, y_sent, seq_len)
        ol = self.get_loss_ol(self.coarse_grained_target, self.sentiment_clue)
        brl = self.get_loss_brl(self.multi_grained_target,
                                self.multi_grained_sentiment)
        return target_loss + sent_loss + beta * ol + gamma * brl

    def crf_decode(self, inputs: Tuple[tf.Tensor]) -> Tuple[tf.Tensor]:
        _, _, seq_len, _, _ = inputs
        multi_grained_target, multi_grained_sentiment = self.call(
            inputs, training=False)
        target_preds, _ = tfa.text.crf_decode(
            potentials=multi_grained_target,
            transition_params=self.target_trans_params,
            sequence_length=seq_len)
        sentiment_preds, _ = tfa.text.crf_decode(
            potentials=multi_grained_sentiment,
            transition_params=self.sentiment_trans_params,
            sequence_length=seq_len)
        return target_preds, sentiment_preds

    def call(self,
             inputs: Tuple[tf.Tensor],
             training: bool = False) -> Tuple[tf.Tensor]:
        '''
        Feed forward

        Args:
            inputs (tuple): (char_ids, word_embedding, sequence_length, target_label, sentiment_label)
            training (bool)

        Returns:
            output (tf.Tensor)
        '''
        if self.input_type == 'ids':
            self.word_embedding = self.word_embedding_layer(inputs[1])
        else:
            self.word_embedding = inputs[1]
        self.char_embedding = self.char_embedding_layer(inputs[0])
        if self.dropout:
            self.char_embedding = self.dropout_layer(self.char_embedding,
                                                     training=training)
            self.word_embedding = self.dropout_layer(self.word_embedding,
                                                     training=training)
        self.coarse_grained_target, self.sentiment_clue, hidden_states = self.coarse_grained_layer(
            self.char_mask_layer(self.char_embedding),
            self.word_mask_layer(self.word_embedding))
        interacted_target, interacted_sentiment = self.interaction_layer(
            self.coarse_grained_target, self.sentiment_clue, hidden_states)
        self.multi_grained_target, self.multi_grained_sentiment = self.fine_grained_layer(
            interacted_target, interacted_sentiment, hidden_states)
        if self.dropout:
            self.multi_grained_target = self.dropout_layer(
                self.multi_grained_target, training=training)
            self.multi_grained_sentiment = self.dropout_layer(
                self.multi_grained_sentiment, training=training)
        return self.multi_grained_target, self.multi_grained_sentiment
