import tensorflow as tf
import nltk
import pandas as pd
from KaggleWord2VecUtility import KaggleWord2VecUtility
from collections import Counter
import sys

class Options(object):
    def __init__(self):
        pass

class DataReader(object):
    def __init__(self):
        pass

    @staticmethod
    def readFile(filename):
        data = pd.read_csv(filename, header=0, delimiter="\t", quoting=3)
        cleanReviews = {}

        print "Cleaning and parsing the set movie reviews...\n"
        for i in xrange( 0, len(data["review"])):
            cleanReviews[data["id"][i]] = KaggleWord2VecUtility.review_to_wordlist(data["review"][i], False)

        return cleanReviews

    @staticmethod
    def buildDict(data):
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
        for i in xrange(min(200,len(wordCount))):
            print ("%s:%d"%(wordCount[i][0],wordCount[i][1]))
        return [ v for (k,v) in wordCount]

class Embedding(object):
    def __init__(self,options):
        self._options = options

    def forward(self,examples,labels):
        opt = self._options
        initWidth = 0.5 / opt.embDim
        emb = tf.Variable(tf.random_uniform([opt.wordSize, opt.embDim],-initWidth,initWidth), name = "emb")
        self._emb = emb

        sm_w_t = tf.Variable(tf.zeros([opt.wordSize, opt.embDim]),name="sm_w_t")
        sm_b = tf.Variable(tf.zeros([opt.wordSize]),name="sm_b")






if __name__ == "__main__":
    filename = sys.argv[1]
    data = DataReader.readFile(filename)
    DataReader.buildDict(data)
