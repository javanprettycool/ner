#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import tensorflow as tf
from data_utils import *
from train import FLAGS
from bi_lstm import BiLSTMModel

def align_data(data):
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]

    data_aligned = dict()
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned


def input_shell(model, sess):
    while True:
        try:
            sentence = raw_input("input >>")
        except NameError:
            sentence = input("input >>")

        words_raw = sentence.strip().split(" ")

        if words_raw == ["exit"]:
            break
        print(words_raw)
        preds = model.predict(words_raw, sess)

        to_print = align_data({"input": words_raw, "output": preds})

        for key, seq in to_print.items():
            print(seq)


def main():
    model_dir = "./runs/1558510358.504878/checkpoints/model"

    train_data, val_data, test_data, vocab_words, vocab_tags, processing_word, _ = load_data(
        FLAGS.train_file, FLAGS.val_file, FLAGS.test_file, FLAGS.words_file, FLAGS.tags_file, FLAGS.chars_file)

    idx_to_tag = {idx: tag for tag, idx in vocab_tags.items()}

    with tf.Session() as sess:
        model = BiLSTMModel(vocab_size=len(vocab_words),
                            num_tags=len(vocab_tags),
                            embedding_size=FLAGS.embedding_size,
                            hidden_size=FLAGS.hidden_size,
                            processing_word=processing_word,
                            idx_to_tag=idx_to_tag)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model_dir)

        input_shell(model, sess)


if __name__ == "__main__":
    main()


