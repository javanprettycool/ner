#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import tensorflow as tf
from data_utils import *
from train import FLAGS
from bi_lstm import BiLSTMModel

def align_data(data):
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]

    named_entities = dict()
    data_aligned = dict()
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    words = data[list(data.keys())[0]]
    preds = data[list(data.keys())[1]]
    start_idx = 0
    start_flag = False
    for i, p in enumerate(preds):
        #print(p, p.startswith("B-"))
        if p.startswith('B-') and not start_flag:
            start_flag = True
            start_idx = i
        elif p == 'O' and start_flag:
            start_flag = False
            word = "".join(words[start_idx:i])
            named_entities[word] = preds[start_idx][2:]
            start_idx = i
        elif p.startswith('B-') and start_flag:
            start_flag = True
            word = "".join(words[start_idx:i])
            named_entities[word] = preds[start_idx][2:]
            start_idx = i


    if start_flag:
        word = "".join(words[start_idx: len(preds)])
        named_entities[word] = preds[start_idx][2:]

    return data_aligned, named_entities


def input_shell(model, sess):
    while True:
        try:
            sentence = raw_input("input >>")
        except NameError:
            sentence = input("input >>")

        words_raw = list(sentence.strip().replace(" ", ""))

        if words_raw == ["exit"]:
            break
        print(words_raw)
        preds = model.predict(words_raw, sess)

        to_print, named_entities = align_data({"input": words_raw, "output": preds})

        for key, seq in to_print.items():
            print(seq)

        print(named_entities)

def main():
    # words_raw = ['我', '来', '自', '广', '东', '梅', '州']
    # preds = ['O', 'O', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'I-LOC']
    # to_print, named_entities = align_data({"input": words_raw, "output": preds})
    # print(to_print, named_entities)
    # exit()


    model_dir = "./runs/1558927314.5551953/checkpoints/model"

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


