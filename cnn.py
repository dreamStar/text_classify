# -*- coding: utf-8 -*-
# coding:utf-8
import tensorflow as tf
from data_reader import DataReader
import math
import sys
import numpy as np

class Cnn_text_classifier(object):
    def __init__(self,sess,config):
        self.config = config
        self.sess = sess
        self.build_graph()

    def build_graph(self):
        config = self.config
        #建立输入和超参数
        self.input_x = tf.placeholder(tf.int32,[None,config['seq_len']], name="input_x")
        self.input_y = tf.placeholder(tf.float32,[None,1], name = "input_y")
        self.drop_out_keep_prob = tf.placeholder(tf.float32,name="drop_out_keep_prob")
        #self.drop_out_keep_prob = tf.constant(config['drop_out_keep_prob']);
        #通过输入向量查找embedding
        with tf.device('/cpu:0'), tf.name_scope("embeding"):
            emb = tf.Variable(tf.random_uniform([config['vocab_size'],config['emb_dim']],-1.0,1.0))
            emb_x = tf.nn.embedding_lookup(emb,self.input_x)
            emb_x_expand = tf.expand_dims(emb_x,-1)

        #进行卷积
        pools = []
        #注意卷积核是有不同高度的,对应着不同的感受窗口
        for filter_size in config['filter_sizes']:
            with tf.name_scope("conv_and_pool_%s" % filter_size):
                #filter_shape的维度是 卷积核高度 * 词嵌入维度 * channel * (该高度)卷积核数目
                filter_shape = [filter_size,config['emb_dim'],1,config['num_filters']]
                conv_w = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name="conv_w")
                conv_b = tf.Variable(tf.constant(0.1,shape=[config['num_filters']]),name="conv_b")
                self.variable_summaries(conv_w,'conv_w')
                self.variable_summaries(conv_b,'conv_b')
                #conv_b = tf.Variable(tf.random_uniform([num_filters],0,0.3),name="conv_b")
                conv = tf.nn.conv2d(
                    emb_x_expand,
                    conv_w,
                    strides=[1,1,1,1],
                    padding="VALID",
                    name="conv"
                )
                conv_biased = tf.nn.bias_add(conv,conv_b)

                h = tf.nn.relu(conv_biased,name="relu")

                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1,config['seq_len'] - filter_size + 1, 1, 1],
                    strides = [1,1,1,1],
                    padding= "VALID",
                    name = "pool"
                )

                pools.append(pooled)

        #合并各tensor
        filter_num_all = config['num_filters'] * len(config['filter_sizes'])
        h_pool = tf.concat(pools,3)
        #展平tensor
        h_pool_flat = tf.reshape(h_pool,[-1,filter_num_all])

        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat,config['drop_out_keep_prob'],name="drop")

        with tf.name_scope("output"):
            out_w = tf.Variable(tf.truncated_normal([filter_num_all,1],stddev=0.1),name="out_w")
            out_b = tf.Variable(tf.constant(0.1,shape=[1]),name="out_b")
            logits = tf.nn.xw_plus_b(h_drop,out_w,out_b,name="score")


        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=self.input_y)
            self.loss = tf.reduce_sum(losses)

        with tf.name_scope("prediction"):
            self.scores = tf.sigmoid(logits,name='scores');
            self.predict = tf.greater(self.scores,self.config['thr'])

        with tf.name_scope("statistics"):
            correct = tf.equal(self.predict,tf.equal(self.input_y,1.0))
            self.accuracy = tf.reduce_mean(tf.cast(correct,'float32'))
            self.auc,_ = tf.metrics.auc(self.input_y,self.scores)
            tf.summary.scalar("accuracy",self.accuracy)
            tf.summary.scalar("ROC",self.auc)

        self.optimizer()
        self.saver = tf.train.Saver(tf.global_variables())
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.config['out_path'] + '/train_summary',
                                      self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.config['out_path'] + '/test_summary',
                                      self.sess.graph)


    def optimizer(self):
        self.global_step = tf.Variable(0,name="global_step",trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads,global_step=self.global_step)

    def train_step(self,x_batch,y_batch):
        feed = {
            self.input_x:x_batch,
            self.input_y:y_batch,
            self.drop_out_keep_prob:self.config['drop_out_keep_prob']
        }
        _,step,loss,scores,summary = self.sess.run([self.train_op,self.global_step,self.loss,self.scores,self.merged],feed)
        print("step %d loss: %f" % (step,loss))
        tmp = np.abs(np.array(scores).reshape(-1) - np.array(y_batch).reshape(-1))
        acc = np.mean(map(lambda x: 1 if x < self.config['thr'] else 0, tmp))
        print("accuracy:%f"%(acc))
        self.train_writer.add_summary(summary,step);

        if step % self.config['check_step'] == 0 :
            path = self.saver.save(self.sess,self.config['out_path']+"/checkpoint",global_step=step)
            print("save checkpoint to " + path)

    def test(self,x,y):
        feed = {
            self.input_x:x,
            self.input_y:y,
            self.drop_out_keep_prob:1.0
        }

        #self.auc,_ = tf.metrics.auc(self.input_y,self.scores)
        step,loss,scores,summary = self.sess.run([self.global_step,self.loss,self.scores,self.merged],feed)
        self.test_writer.add_summary(summary,step)
        #self.train_writer.add_summary(auc,step)
        print("test loss:%f"%(loss))



    def train(self,train_set,test_set):
        print("train_set sum:%d"%len(train_set['x']))
        print("valid_set sum:%d"%len(valid_set['x']))

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        batches = DataReader.batch_iter(train_set['x'],train_set['y'],self.config['batch_size'],self.config['epoch'])
        cnt = 0
        for (batch_x,batch_y) in batches:
            cnt += 1
            self.train_step(batch_x,batch_y)
            if cnt % self.config['test_step'] == 0:
                self.test(test_set['x'],test_set['y'])
        [step] = self.sess.run([self.global_step])
        path = self.saver.save(self.sess,self.config['out_path']+"/checkpoint",global_step = step)

    def variable_summaries(self,var,name):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name+'_mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar(name+'_stddev', stddev)
        tf.summary.scalar(name+'_max', tf.reduce_max(var))
        tf.summary.scalar(name+'_min', tf.reduce_min(var))
        tf.summary.histogram(name+'_histogram', var)

def read_data(train_filename):
    data = DataReader.read_file(train_filename,True)
    id2word,word2id = DataReader.build_dict(data['samples'],0.8);

    max_len = max([len(x) for x in data['samples'].values()])

    train_data , test_data = DataReader.split_data(data['samples'],data['labels'])

    print("word sample: ")
    print(train_data['x'][0])

    train_data['x'] = DataReader.parse2index(train_data['x'],word2id,max_len)
    test_data['x'] = DataReader.parse2index(test_data['x'],word2id,max_len)
    print("id sample: ")
    print(train_data['x'][0])
    #test_data = DataReader.read_file(test_filename,True)
    #test_data = DataReader.parse2index(test_data,word2id)


    #print("train_data:")
    #print(train_data)

    return train_data,test_data,id2word,word2id




if __name__ == "__main__":
    #train_filename = sys.argv[1]
    #test_filename = sys.argv[2]
    #train_set,test_set,id2word,word2id = read_data(train_filename)
    obj_prefix = sys.argv[1]

    word2id,id2word = DataReader.load_dict(obj_prefix)
    train_set,valid_set = DataReader.load_data(obj_prefix)

    #seq_len = max([len(x) for x in train_set['x']])

    config = {
        'drop_out_keep_prob' : 0.5,
        'emb_dim' : 128,
        'vocab_size' : len(word2id),
        'seq_len' : 500,
        'filter_sizes': [3,4,5],
        'num_filters': 128,
        'epoch':200,
        'batch_size':64,
        'thr':0.5,
        'out_path':'./out',
        'check_step':1000,
        'test_step' : 1000
    }

    cnn = Cnn_text_classifier(tf.Session(),config)
    cnn.train(train_set,valid_set)
