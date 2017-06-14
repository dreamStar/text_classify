import tensorflow as tf
from data_reader import DataReader
import math
import sys
import numpy as np

def eval(ckpt,input_data):
    ret = []
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph(ckpt+".meta")
            saver.restore(ckpt)

            input_x = graph.get_operation_by_name("input_x").outputs[0]
            drop_out_keep_prob = graph.get_operation_by_name("drop_out_keep_prob").outputs[0]
            scores = graph.get_operation_by_name("prediction/scores").outputs[0]

            batches = DataReader.batch_iter(input_data)
            for (input_batch) in batches:
                feed = {
                    "input_x":input_batch,
                    "drop_out_keep_prob": 1
                }
                [predict_scores] = sess.run([scores],feed_dict=feed)
                ret = np.concatenate([ret,predict_scores])
    return ret
