import tensorflow as tf
from data_reader import DataReader
import math
import sys
import numpy as np
import cPickle as pickle

def eval(ckpt,input_data):
    ret = np.array([])
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph(ckpt+".meta")
            saver.restore(sess,ckpt)

            input_x = graph.get_operation_by_name("input_x").outputs[0]
            drop_out_keep_prob = graph.get_operation_by_name("drop_out_keep_prob").outputs[0]
            scores = graph.get_operation_by_name("prediction/scores").outputs[0]

            batches = DataReader.batch_iter(input_data,shuffle=False)
            for (input_batch) in batches:
                feed = {
                    input_x:input_batch,
                    drop_out_keep_prob: 1
                }
                [predict_scores] = sess.run([scores],feed_dict=feed)

                ret = np.concatenate([ret,np.array(predict_scores).flatten()])
    return ret

def print_out(out_file,id,ret):
    with open(out_file,'w') as f:
        f.write('"id","sentiment"\n')
        for i in xrange(len(ret)):
            f.write('%s,%f\n'%(id[i],ret[i]))

if __name__ == "__main__":
    input_prefix = sys.argv[1]
    ckpt = sys.argv[2]
    out_file = sys.argv[3]

    input_data = pickle.load(open(input_prefix+'_test_data','r'))
    data_id = pickle.load(open(input_prefix+'_test_id','r'))
    ckpt_file = tf.train.latest_checkpoint(ckpt)
    ret = eval(ckpt_file,input_data)
    print("ret len:%d"%len(ret))
    print("some ret:")
    for i in xrange(20):
        print("no.%d ret:%d"%(i,ret[i]))
    print_out(out_file,data_id,ret)
