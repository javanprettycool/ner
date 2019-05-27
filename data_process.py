#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import re
from gensim.models import word2vec

WRITE_INDEX = 5
DEV_INDEX = [3]
TEST_INDEX = [4]


def word2char():
    p = '/\w{1,2}\s{2}'
    p_n = '\]\w+'
    # 人名
    p_name = '\w+/nr(\s{2}\w+/nr)?'
    # 地名
    p_loc = '\[(\w+/\w+(\s{2})?)+\]ns | \w+/ns(\s{2}\w+/ns)?'
    # 机构
    p_orgn = '\[(\w+/\w+(\s{2})?)+\]nt | \w+/nt(\s{2}\w+/nt)?'
    # 时间
    p_time = '\w+/t(\s{2}\w+/t)?'

    train_file = open('./data/train_data.txt', 'w', encoding='utf-8')
    val_file = open('./data/val_data.txt', 'w', encoding='utf-8')
    test_file = open('./data/test_data.txt', 'w', encoding='utf-8')
    i = 0

    with open('./corpus/98corpus.txt', encoding='utf-8') as f:
        for line in f:
            raw_line = line

            line = line.replace('[', '')
            line = re.sub(p_n, '', line)
            rs = re.split(p, line)[1:-1]
            s_list = [c for s in rs for c in s]
            l_list = ['O' for _ in range(len(s_list))]

            # 处理机构实体
            r_orgn = re.finditer(p_orgn, raw_line)
            for o in r_orgn:
                orgns = o.group().replace('[', '').replace(']', '')
                orgns = re.sub('/\w+', '', orgns).replace('  ', '').strip()
                s_list_process(s_list, l_list, orgns, 'B-ORG', 'I-ORG')

            # 处理人名实体
            r_names = re.finditer(p_name, line)
            for m in r_names:
                name = m.group().replace('/nr', '').replace('  ', '').strip()
                s_list_process(s_list, l_list, name, 'B-PER', 'I-PER')

            # 处理地名实体
            r_loc = re.finditer(p_loc, raw_line)
            for l in r_loc:
                loc = l.group().replace('[', '').replace(']', '')
                loc = re.sub('/\w+', '', loc).replace('  ', '').strip()
                s_list_process(s_list, l_list, loc, 'B-LOC', 'I-LOC')

            # 处理时间
            r_time = re.finditer(p_time, raw_line)
            for t in r_time:
                time = t.group().replace('[', '').replace(']', '')
                time = re.sub('/\w+', '', time).replace('  ', '').strip()
                print(time)
                s_list_process(s_list, l_list, time, 'B-DATE', 'I-DATE')

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


def s_list_process(s_list, l_list, name, tag_start, tag_end):
    for index in range(len(s_list) - len(name) + 1):
        if s_list[index] == name[0]:
            for n_index in range(1, len(name)):
                if s_list[index + n_index] != name[n_index]:
                    break
            else:
                l_list[index] = tag_start
                for i in range(1, len(name)):
                    l_list[index + i] = tag_end


if __name__ == '__main__':
    word2char()
    #word2char_embedding()





