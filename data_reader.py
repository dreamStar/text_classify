# -*- coding: utf-8 -*-
# coding:utf-8

import tensorflow as tf
import nltk
import pandas as pd
from KaggleWord2VecUtility import KaggleWord2VecUtility
from collections import Counter
import sys
import math
import random
import numpy as np

unknown_sym = "<unknown>"
padding_sym = "<pading>"

class Options(object):
    def __init__(self):
        pass

class DataReader(object):
    def __init__(self):
        pass

    #将kaggle提供的文本数据读取并解析为id:text的字典
    @staticmethod
    def read_file(filename,labeled):
        data = pd.read_csv(filename, header=0, delimiter="\t", quoting=3)
        clean_reviews = {}
        labels = {}

        print "Cleaning and parsing the set movie reviews...\n"
        for i in xrange( 0, len(data["review"])):
            clean_reviews[data["id"][i]] = KaggleWord2VecUtility.review_to_wordlist(data["review"][i], False)
            if(labeled):
                labels[data["id"][i]] = data["sentiment"][i]

        ret = {'samples':clean_reviews}
        if(labeled):
            ret["labels"] = labels

        return ret

    #建立id与word对应关系
    @staticmethod
    def build_dict(data,vocab_rate):
        #print("data:")
        #print(data)
        lemmatizer = nltk.stem.WordNetLemmatizer()

        lex = []
        for words in data.values():
            #lex.extend(words)
            lex.extend([lemmatizer.lemmatize(word) for word in words])

        wordCount = Counter(lex)

        wordNum = len(lex)
        #print("lex:")
        #print(lex)
        #print("wordCount:")
        #print(wordCount)
        uniWordNum = len(wordCount)

        wordCount = sorted(wordCount.items(),key=lambda d: d[1],reverse=True)


        print ("word num : %d\nunique word : %d\n"%(wordNum,uniWordNum))
        print ("some leading words:")
        for i in xrange(min(20,len(wordCount))):
            print ("%s:%d"%(wordCount[i][0],wordCount[i][1]))

        vocab_size = int(math.floor(vocab_rate * uniWordNum))

        words = [ k for (k,v) in wordCount]
        words = words[0:vocab_size]
        # print("vocab_size:%d"%vocab_size)
        # print("words:")
        # print(words)
        print ("word sum : %d "%(len(words)+2))
        id2word = {0:padding_sym,1:unknown_sym}
        word2id = {padding_sym:0,unknown_sym:1}
        for i in xrange(len(words)):
            id2word[i+2] = words[i]
            word2id[words[i]] = i+2

        return id2word,word2id

    #data_x是包含所有句子单词列表的列表,返回是这些词替换为对应id的结果.所有句子都统一到同一个长度
    @staticmethod
    def parse2index(data_x,word2id,sentence_len=0):
        ret = []
        for sentence in data_x:
            id_array = map(lambda w: word2id[w] if w in word2id else word2id[unknown_sym], sentence)
            if sentence_len > 0 :
                id_array = id_array + [word2id[padding_sym]] * (sentence_len - len(id_array)) if sentence_len > len(id_array) else id_array[0:sentence_len]
            ret.append(id_array)

        return ret


    #将数据data_set和标签label_set按照指定比例分割为测试集和训练集
    @staticmethod
    def split_data(data_set,label_set,train_rate=0.7,shuffle=True):
        sample_num = len(data_set)
        train_num = int(sample_num * train_rate)
        test_num = sample_num - train_num

        index = range(sample_num)
        if shuffle:
            random.shuffle(index)
        train_index = index[0:train_num]
        test_index = index[train_num:-1]

        train_x = [data_set.values()[i] for i in train_index]
        test_x = [data_set.values()[i] for i in test_index]

        train_y = [[label_set.values()[i]] for i in train_index]
        test_y = [[label_set.values()[i]] for i in test_index]

        train_id = [data_set.keys()[i] for i in train_index]
        test_id = [data_set.keys()[i] for i in test_index]

        return {'x':train_x,'y':train_y,'id':train_id},{'x':test_x,'y':test_y,'id':test_id}

    @staticmethod
    def batch_iter(data_x,data_y, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        data_size = len(data_x)
        num_batches_per_epoch = int((len(data_x)-1)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_x = data_x[shuffle_indices]
                shuffled_y = data_y[shuffle_indices]
            else:
                shuffled_x = data_x
                shuffled_y = data_y
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield (shuffled_x[start_index:end_index],shuffled_y[start_index:end_index])


if __name__ == "__main__":
    filename = sys.argv[1]
    data = DataReader.readFile(filename)
    DataReader.buildDict(data)
