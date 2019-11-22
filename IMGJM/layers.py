# Copyright (c) 2019 seanchang
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from abc import ABCMeta, abstractmethod
from typing import Tuple, Union
import numpy as np
import tensorflow as tf


class BaseLayer(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, x):
        return NotImplemented


class Gate(BaseLayer):
    '''
    Gate layer

    Attributes:
        dim_a (int): dimension of a.
        dim_b (int): dimension of b.
    '''
    def __init__(self, dim_a: int, dim_b: int, *args, **kwargs):
        with tf.variable_scope('Gate_Variables', reuse=tf.AUTO_REUSE):
            self.W_g = tf.get_variable('W_g',
                                       shape=[dim_b, dim_b],
                                       dtype=tf.float32)
            self.W_trans = tf.get_variable('W_trans',
                                           shape=[dim_b, dim_a],
                                           dtype=tf.float32)

    def __call__(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        '''
        Args:
            a (tf.Tensor)
            b (tf.Tensor)
        
        Returns:
            output (tf.Tensor)
        '''
        with tf.name_scope('Gate'):
            g_ = tf.matmul(a, self.W_trans, transpose_b=True)
            g = tf.nn.sigmoid(tf.matmul(g_, self.W_g))
            output = g * tf.matmul(a, self.W_trans,
                                   transpose_b=True) + (1 - g) * b
            return output


class GloveEmbedding(BaseLayer):
    def __call__(self, inputs: Union[np.ndarray, tf.Tensor], embedding_placeholder: tf.Tensor) -> tf.Tensor:
        word_embeddings = tf.nn.embedding_lookup(embedding_placeholder,
                                                 inputs)
        return word_embeddings


class CharEmbedding(BaseLayer):
    def __init__(self,
                 vocab_size: int,
                 embedding_size: int = 50,
                 window_size: int = 3,
                 filter_nums: int = 50,
                 *args,
                 **kwargs):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.filter_nums = filter_nums
        embedding_weights_ = np.random.uniform(-0.1,
                                               0.1,
                                               size=(vocab_size,
                                                     embedding_size))
        pad = np.zeros(shape=(1, embedding_size))
        unk = np.random.uniform(-0.1, 0.1, size=(1, embedding_size))
        embedding_weights = np.concatenate((embedding_weights_, pad, unk),
                                           axis=0)
        with tf.variable_scope('CharEmbedding_Variables', reuse=tf.AUTO_REUSE):
            self.embedding_weights = tf.get_variable(
                'embedding_weights',
                shape=[self.vocab_size, self.embedding_size],
                initializer=tf.initializers.constant(embedding_weights),
                dtype=tf.float32)
            self.conv_weights = tf.get_variable('conv_weights',
                                                shape=[
                                                    self.window_size,
                                                    self.embedding_size,
                                                    self.filter_nums
                                                ])
            self.conv_bias = tf.get_variable('conv_bias',
                                             shape=[self.filter_nums])

    def __call__(self, inputs: Union[np.ndarray, tf.Tensor]) -> tf.Tensor:
        with tf.name_scope('CharEmbedding'):
            char_embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                    inputs,
                                                    name='char_embedding')
            sp = char_embedding.shape
            char_embedding_ = tf.reshape(char_embedding,
                                         shape=[sp[0] * sp[1], sp[2], -1])
            cnn_output = tf.nn.conv1d(char_embedding_,
                                      filters=self.conv_weights,
                                      stride=1,
                                      padding='VALID')
            cnn_output = tf.nn.relu(cnn_output + self.conv_bias)
            cnn_output = tf.nn.max_pool1d(cnn_output,
                                          ksize=[self.window_size, 1, 1],
                                          strides=[1, 1, 1],
                                          padding='VALID')
            cnn_output = tf.reshape(cnn_output, shape=(sp[0], sp[1], sp[-1]))
            return cnn_output


class CoarseGrainedLayer(BaseLayer):
    '''
    Coarse-grained tagging layer
    d_h = 2(d_w + d_ch) = 700

    Attributes:
        hidden_nums (int): cell units nums
    '''
    def __init__(self, hidden_nums: int = 700, *args, **kwargs):
        with tf.variable_scope('CoarseGrainedLayer_Variables',
                               reuse=tf.AUTO_REUSE):
            self.cell_fw = tf.nn.rnn_cell.GRUCell(hidden_nums / 2)
            self.cell_bw = tf.nn.rnn_cell.GRUCell(hidden_nums / 2)
            self.W_z = tf.get_variable('W_z',
                                       shape=[2, hidden_nums],
                                       dtype=tf.float32)
            self.W_q = tf.get_variable('W_q',
                                       shape=[2, hidden_nums],
                                       dtype=tf.float32)

    def __call__(self, char_embedding: tf.Tensor, word_embedding: tf.Tensor,
                 sequence_length: tf.Tensor) -> Tuple[tf.Tensor]:
        '''
        Args:
            char_embedding (tf.Tensor)
            word_embedding (tf.Tensor)
            sequence_length (tf.Tensor)

        Returns:
            z_T (tf.Tensor): coarse-grained target
            z_S (tf.Tensor): sentiment clue
            hidden_states (tf.Tensor): bi-GRU hidden states
        '''
        with tf.name_scope('CoarseGrainedLayer'):
            word_representation = tf.concat([char_embedding, word_embedding],
                                            axis=-1)
            (hidden_states, final_states) = tf.nn.bidirectional_dynamic_rnn(
                self.cell_fw,
                self.cell_bw,
                inputs=word_representation,
                sequence_length=sequence_length,
                dtype=tf.float32)
            hidden_states = tf.concat(hidden_states, axis=-1)
            z_T = tf.nn.softmax(
                tf.matmul(hidden_states, self.W_z, transpose_b=True))
            z_S = tf.nn.softmax(
                tf.matmul(hidden_states, self.W_q, transpose_b=True))
            return (z_T, z_S, hidden_states)


class Interaction(BaseLayer):
    '''
    Interaction layer
    d_h = 2(d_w + d_ch) = 700

    Attributes:
        hidden_nums (int)
        C_tar (int)
        C_sent (int)
    '''
    def __init__(self,
                 hidden_nums: int = 700,
                 C_tar: int = 5,
                 C_sent: int = 7,
                 *args,
                 **kwargs):
        with tf.variable_scope('Interaction_Variables', reuse=tf.AUTO_REUSE):
            self.W_att = tf.get_variable('W_att',
                                         shape=[hidden_nums, hidden_nums],
                                         dtype=tf.float32)
            self.W_l = tf.get_variable('W_l', shape=[C_tar, 2])
            self.W_r = tf.get_variable('W_r', shape=[C_sent, hidden_nums])
            self.gate_T = Gate(dim_a=2, dim_b=2)
            self.gate_S = Gate(dim_a=2, dim_b=2)

    def __call__(self, coarse_grained_target: tf.Tensor,
                 sentiment_clue: tf.Tensor,
                 hidden_states: tf.Tensor) -> Tuple[tf.Tensor]:
        '''
        Args:
            coarse_grained_target (tf.Tensor)
            sentiment_clue (tf.Tensor)
            hidden_states (tf.Tensor)
        
        Returns:
            l_T (tf.Tensor)
            l_S (tf.Tensor)
        '''
        with tf.name_scope('Interaction'):
            with tf.name_scope('Sentiment_to_target'):
                U = tf.nn.tanh(
                    tf.matmul(tf.matmul(hidden_states, self.W_att),
                              hidden_states,
                              transpose_b=True))
                A = tf.nn.softmax(U)
                sa_T = tf.matmul(A, sentiment_clue)
                l_T = tf.matmul(self.gate_T(sentiment_clue, sa_T),
                                self.W_l,
                                transpose_b=True)
            with tf.name_scope('Target_to_sentiment'):
                r = tf.matmul(A, hidden_states)
                att_S = tf.nn.relu(tf.matmul(r, self.W_r, transpose_b=True))
                l_S = self.gate_S(l_T, att_S)
            return (l_T, l_S)


class FineGrainedLayer(BaseLayer):
    def __init__(self,
                 hidden_nums: int = 700,
                 C_tar: int = 5,
                 C_sent: int = 7,
                 *args,
                 **kwargs):
        with tf.variable_scope('FineGrainedLayer_Variables',
                               reuse=tf.AUTO_REUSE):
            self.cell_fw = tf.nn.rnn_cell.GRUCell(hidden_nums / 2)
            self.cell_bw = tf.nn.rnn_cell.GRUCell(hidden_nums / 2)
            self.W_ft = tf.get_variable('W_ft',
                                        shape=[C_tar, hidden_nums],
                                        dtype=tf.float32)
            self.W_fs = tf.get_variable('W_fs',
                                        shape=[C_sent, hidden_nums],
                                        dtype=tf.float32)
            self.gate_T = Gate(dim_a=C_tar, dim_b=C_tar)
            self.gate_S = Gate(dim_a=C_sent, dim_b=C_sent)

    def __call__(self, interacted_target: tf.Tensor,
                 interacted_sentiment: tf.Tensor, hidden_states: tf.Tensor,
                 sequence_length: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('FineGrainedLayer'):
            (hidden_states_, final_outputs) = tf.nn.bidirectional_dynamic_rnn(
                self.cell_fw,
                self.cell_bw,
                inputs=hidden_states,
                sequence_length=sequence_length,
                dtype=tf.float32)
            f_T = tf.matmul(hidden_states_, self.W_ft, transpose_b=True)
            f_S = tf.matmul(hidden_states_, self.W_fs, transpose_b=True)
            o_T = self.gate_T(interacted_target, f_T)
            o_S = self.gate_S(interacted_sentiment, f_S)
            return (o_T, o_S)