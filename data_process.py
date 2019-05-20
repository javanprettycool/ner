#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import re
from gensim.models import word2vec

WRITE_INDEX = 5
DEV_INDEX = [3]
TEST_INDEX = [4]


def word2char():
    p = '/\w{1,2}\s{2}'
    p_name = '\w+/nr(\s{2}\w+/nr)?'
    p_n = '\]\w+'
    train_file = open('./data/train_data.txt', 'w', encoding='utf-8')
    val_file = open('./data/val_data.txt', 'w', encoding='utf-8')
    test_file = open('./data/test_data.txt', 'w', encoding='utf-8')
    i = 0

    with open('./corpus/98corpus.txt', encoding='utf-8') as f:
        for line in f:
            line = line.replace('[', '')
            line = re.sub(p_n, '', line)
            rs = re.split(p, line)[1:-1]
            s_list = [c for s in rs for c in s]
            l_list = ['O' for _ in range(len(s_list))]
            r_names = re.finditer(p_name, line)
            for m in r_names:
                name = m.group().replace('/nr', '').replace('  ', '')
                s_list_process(s_list, l_list, name)
            if i % WRITE_INDEX in DEV_INDEX:
                wf = val_file
            elif i % WRITE_INDEX in TEST_INDEX:
                wf = test_file
            else:
                wf = train_file

            for index in range(len(s_list)):
                wf.write(s_list[index] + ' ' + l_list[index] + '\n')

            wf.write('\n')
            i += 1


def word2char_embedding():
    p = '/\w{1,2}\s{2}'
    p_n = '\]\w+'
    word2char_file = open('./data/word2char.txt', 'w', encoding='utf-8')
    with open('./corpus/98corpus.txt', encoding='utf-8') as f:
        for line in f:
            line = line.replace('[', '')
            line = re.sub(p_n, '', line)
            rs = re.split(p, line)[1:-1]
            s_list = [c for s in rs for c in s]
            if s_list:
                for index in range(len(s_list) - 1):
                    word2char_file.write(s_list[index] + ' ')
                word2char_file.write(s_list[len(s_list) - 1])
                word2char_file.write('\n')
    word2char_file.close()
    sentences = word2vec.Text8Corpus('./data/word2char.txt')
    model = word2vec.Word2Vec(sentences, size=300)
    model.wv.save_word2vec_format('./data/word2vec.txt', binary=False)
    #model.save()


def s_list_process(s_list, l_list, name):
    for index in range(len(s_list) - len(name) + 1):
        if s_list[index] == name[0]:
            for n_index in range(1, len(name)):
                if s_list[index + n_index] != name[n_index]:
                    break
            else:
                l_list[index] = 'B-PER'
                for i in range(1, len(name)):
                    l_list[index + i] = 'I-PER'


if __name__ == '__main__':
    #word2char()
    word2char_embedding()





