#
# Util class for help training.
#

import numpy as np
from numpy import arange
import tensorflow as tf
from tensorflow import(reduce_mean, matmul)

from six.moves import cPickle as pickle


class TrainingHelper:
    def __init__(self, each_object_size_width=28, each_object_size_height=28):
        """
        Constructor 
        """
        self.each_object_size_width = each_object_size_width
        self.each_object_size_height = each_object_size_height

    def save_pickle(self, pickle_fullname, data_to_save):
        """
        Save data_to_save to a pickle.
        """
        try:
            with open(pickle_fullname, "wb") as f:
                pickle.dump(data_to_save, f, pickle.HIGHEST_PROTOCOL)
                return f
        except Exception as e:
            print("Unable to read {}: {}".format(pickle_fullname,  e))
            raise

    def flat_dataset_labels(self, dataset, labels, count_classes):
        """
        Flat dataset, labels to 2-D arrays.
        """
        ds = dataset.reshape((-1, self.each_object_size_width *
                              self.each_object_size_height)).astype(np.float32)
        lb = (arange(count_classes) == labels[:, None]).astype(np.float32)
        return ds, lb

    def accuracy(self,  predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

    def activation(self, x, weights, biases):
        """
        Define math computation, the logits.
        """
        return matmul(x, weights) + biases

    def loss_optimizer(self, y, activation, train_learning_rate, beta, weights):
        loss = reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=activation))
        # Loss function using L2 Regularization
        reg = tf.nn.l2_loss(weights)
        loss = reduce_mean(loss + beta * reg)

        optimizer = tf.train.GradientDescentOptimizer(
            train_learning_rate).minimize(loss)

        return loss, optimizer
