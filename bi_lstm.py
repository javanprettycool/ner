# encoding=utf8

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf


class BiLSTMModel(object):
    def __init__(
            self, vocab_size, num_tags, embedding_size, hidden_size, learning_rate=1e-3):

        self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        def lstm_cell():
            cell = rnn.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
            return rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)

        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            self.embedding = tf.get_variable("embedding", [vocab_size, embedding_size])
            self.embedded_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope("bi-lstm"):
            self.dynamic_length = tf.cast(tf.reduce_sum(tf.sign(self.input_x)), tf.int32)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell(),
                                                                        lstm_cell(),
                                                                        self.embedded_inputs,
                                                                        seq_length=self.dynamic_length)

        with tf.name_scope("output"):
            output = tf.concat([output_fw, output_bw], axis=-1)
            seq_length = tf.shape(output)[1]
            output = tf.reshape(output, [-1, hidden_size * 2])

        with tf.name_scope("optimize"):
            w = tf.Variable(tf.truncated_normal([hidden_size * 2, num_tags]), name="w")
            b = tf.constant(0.1, [num_tags], name="b")
            self.scores = tf.nn.xw_plus_b(output, w, b, name="score")
            self.predictions = tf.reshape(self.scores, [-1, seq_length, num_tags])
            log_likehood, self.transition_params = crf.crf_log_likelihood(self.predictions, self.input_y, self.dynamic_length)
            self.loss = tf.reduce_mean(-log_likehood)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = optimizer.minimize(self.loss)
            #grads_and_vars = optimizer.compute_gradients(self.loss)
            #optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        with tf.name_scope("accuracy"):
            output_seq, _ = crf.viterbi_decode(
                self.predictions[:self.dynamic_length],
                self.transition_params)
            label = self.input_y[:self.dynamic_length]
            correct_predictions = tf.equal(output_seq, label)
            self.acc = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def predict_batch(self, words):


        pass

    def predict(self, words_raw):
        pass