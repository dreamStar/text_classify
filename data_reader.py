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
import cPickle as pickle
import os

unknown_sym = "UNK"
padding_sym = "UNK"

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
        print ("word dict sum : %d "%(len(words)+2))
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
        unk_cnt = 0
        word_cnt = 0
        print("sentence_len:%d"%sentence_len)
        for sentence in data_x:
            id_array = map(lambda w: word2id[w] if w in word2id else word2id[unknown_sym], sentence)
            for w in sentence:
                word_cnt += 1
                if not w in word2id:
                    unk_cnt += 1
            if sentence_len > 0 :
                id_array = id_array + [word2id[padding_sym]] * (sentence_len - len(id_array)) if sentence_len > len(id_array) else id_array[0:sentence_len]
            ret.append(id_array)
        print("%d unknown words in %d words"%(unk_cnt,word_cnt))

        return ret


    #将数据data_set和标签label_set按照指定比例分割为测试集和训练集
    @staticmethod
    def split_data(data_set,label_set,train_rate=0.9,shuffle=True):
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
    def batch_iter(data_x,data_y=None, batch_size=64, num_epochs=1, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data_x = np.array(data_x)
        if not data_y is None:
            data_y = np.array(data_y)
        data_size = len(data_x)
        num_batches_per_epoch = int((len(data_x)-1)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_x = data_x[shuffle_indices]
                if not data_y is None:
                    shuffled_y = data_y[shuffle_indices]
            else:
                shuffled_x = data_x
                if not data_y is None:
                    shuffled_y = data_y
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                if not data_y is None:
                    yield (shuffled_x[start_index:end_index],shuffled_y[start_index:end_index])
                else:
                    yield (shuffled_x[start_index:end_index])

    @staticmethod
    def static_sentence_len(sentences):
        bin_max = 100
        bins = [0]*bin_max
        sum = 0
        for sen in sentences:
            sum += len(sen)
            index = int(math.floor(len(sen)/10))

            bins[index if index < bin_max else bin_max-1] += 1
        avg = sum / len(sentences)
        print("sentences sum:%d"%len(sentences))
        print("sentences avg len:" + str(avg))
        for i in xrange(bin_max-1):
            print("len < %d : %d" %((i+1)*10,bins[i]))
        print("len >= %d : %d" %((bin_max-1)*10,bins[bin_max-1]))

    @staticmethod
    def static_pn_rate(labels):
        sum = len(labels)
        p = np.sum(labels)
        print ("%d postive samples of %d samples (%f)" %(p,sum,p/sum))

    @staticmethod
    def load_dict(file_prefix):
        #emb = pickle.load(file_prefix+'_emb')
        word2id = pickle.load(open(file_prefix+'_word2id','r'))
        id2word = pickle.load(open(file_prefix+'_id2word','r'))
        return word2id,id2word

    @staticmethod
    def read_data(train_filename,out_prefix,seq_len,input_prefix=None):
        print("params:")
        print(train_filename)
        print(out_prefix)
        print(seq_len)
        print(input_prefix)
        data = DataReader.read_file(train_filename,True)
        if input_prefix is None:
            id2word,word2id = DataReader.build_dict(data['samples'],0.8);
        else:
            id2word = pickle.load(open(input_prefix+'_id2word','r'))
            word2id = pickle.load(open(input_prefix+'_word2id','r'))

        DataReader.static_sentence_len(data['samples'].values())
        DataReader.static_pn_rate(data['labels'].values())

        #max_len = max([len(x) for x in data['samples'].values()])
        max_len = int(seq_len)

        train_data , test_data = DataReader.split_data(data['samples'],data['labels'])

        print("word sample: ")
        print(train_data['x'][0])

        train_data['x'] = DataReader.parse2index(train_data['x'],word2id,max_len)
        test_data['x'] = DataReader.parse2index(test_data['x'],word2id,max_len)
        print("id sample: ")
        print(train_data['x'][0])
        #test_data = DataReader.read_file(test_filename,True)
        #test_data = DataReader.parse2index(test_data,word2id)
        pickle.dump(train_data,open(out_prefix+'_train_data','w'),True)
        pickle.dump(test_data,open(out_prefix+'_valid_data','w'),True)
        pickle.dump(id2word,open(out_prefix+'_id2word','w'),True)
        pickle.dump(word2id,open(out_prefix+'_word2id','w'),True)

        #print("train_data:")
        #print(train_data)

        return train_data,test_data,id2word,word2id

    @staticmethod
    def read_test_data(test_filename,dict_prifix,out_prefix,seq_len):
        data = DataReader.read_file(test_filename,False)
        #id2word = pickle.load(open(dict_prifix+'_id2word','r'))
        word2id = pickle.load(open(dict_prifix+'_word2id','r'))


        DataReader.static_sentence_len(data['samples'].values())

        test_x = data['samples'].values()
        max_len = int(seq_len)

        test_data = DataReader.parse2index(test_x,word2id,max_len)
        pickle.dump(test_data,open(out_prefix+'_test_data','w'),True)
        pickle.dump(data['samples'].keys(),open(out_prefix+'_test_id','w'),True)
        #max_len = max([len(x) for x in data['samples'].values()])


        #print("train_data:")
        #print(train_data)

        return test_data,word2id

    @staticmethod
    def load_word2vec(prefix):
        emb = pickle.load(open(prefix+'_emb','r'))
        return emb

    @staticmethod
    def load_data(file_prefix):
        train_data = pickle.load(open(file_prefix+'_train_data','r'))
        test_data = pickle.load(open(file_prefix+'_valid_data','r'))
        return train_data,test_data


if __name__ == "__main__":
    opt = sys.argv[1]
    if opt == 'train':
        filename = sys.argv[2]
        out_prefix = sys.argv[3]
        seq_len = sys.argv[4]
        dict_prefix = None
        if len(sys.argv) == 6:
            dict_prefix = sys.argv[5]
        data = DataReader.read_data(filename,out_prefix,seq_len,dict_prefix)
    elif opt == 'test':
        filename = sys.argv[2]
        out_prefix = sys.argv[3]
        dict_prifix = sys.argv[4]
        seq_len = sys.argv[5]
        data = DataReader.read_test_data(filename,dict_prifix,out_prefix,seq_len)
