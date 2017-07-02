import os
import sys
import tensorflow as tf
import logging
import numpy as np
from task1.model import Model
from task1.model import test_case
import argparse

def train_run():
    logging.info('Training start')
    with tf.Graph().as_default():
        #with tf.device("/gpu:" + str(args.gpu_num)):
        classifier = Model()
        saver = tf.train.Saver()
        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:

            best_accuracy = 0
            best_val_epoch = 0
            sess.run(tf.global_variables_initializer())

            train_data_x = np.array(classifier.train_data_x)
            train_data_y = np.array(classifier.train_data_y)
            train_data_len = np.array(classifier.train_data_len)

            val_data_x = np.array(classifier.val_data_x)
            val_data_y = np.array(classifier.val_data_y)
            val_data_len = np.array(classifier.val_data_len)

            for epoch in range(classifier.config.max_epochs):
                print("="*20+"Epoch ", epoch, "="*20)
                loss = classifier.run_epoch(sess, train_data_x, train_data_y, train_data_len)
                print()
                print ("Mean loss in this epoch is: ", loss)
                logging.info("Mean loss in {}th epoch is: {}".format(epoch, loss) )
                print ('='*50)

                if args.debug_enable:
                    test_case(sess, classifier, train_data_x, train_data_y, train_data_len, onset='TRAINING')
                val_accuracy, loss = test_case(sess, classifier, val_data_x, val_data_y, val_data_len, onset='VALIDATION')

                if best_accuracy < val_accuracy:
                    best_accuracy = val_accuracy
                    best_val_epoch = epoch
                    if not os.path.exists(classifier.weight_Path):
                        os.makedirs(classifier.weight_Path)

                    saver.save(sess, classifier.weight_Path+'/classifier.weights')
                if epoch - best_val_epoch > classifier.config.early_stopping:
                    logging.info("Normal Early stop")
                    break
    logging.info("Training complete")
def main(_):
    if not os.path.exists(args.weight_path):
        os.makedirs(args.weight_path)
    logFile = args.weight_path+'/run.log'

    try:
        os.remove(logFile)
    except OSError:
        pass
    logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
    train_run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training options")
    parser.add_argument('--weight-path', action='store', dest='weight_path', default="./savings/save01")
    parser.add_argument('--debug-enable', action='store_true', dest='debug_enable', default=False)
    parser.add_argument('--gpu-num', action='store', dest='gpu_num', default=0, type=int)
    args = parser.parse_args()
if __name__ == '__main__':
    tf.app.run()