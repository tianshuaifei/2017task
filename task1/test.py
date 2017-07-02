import os
import sys
import tensorflow as tf
import logging
import numpy as np
from task1.model import Model
from task1.model import test_case
import argparse


def test_run():

    with tf.Graph().as_default():
       # with tf.device("/gpu:" + str(args.gpu_num)):   #gpu_num options
        classifier = Model()
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, classifier.weight_Path+'/classifier.weights')

            test_data_x = np.array(classifier.test_data_x)
            test_data_y = np.array(classifier.test_data_y)
            test_data_len = np.array(classifier.test_data_len)

            accu, loss = test_case(sess, classifier, test_data_x, test_data_y, test_data_len, onset='TEST')

def main(_):
    logFile = "./savings/save01"+'/run.log'

    try:
        os.remove(logFile)
    except OSError:
        pass
    logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
    test_run()
if __name__ == '__main__':

    tf.app.run()