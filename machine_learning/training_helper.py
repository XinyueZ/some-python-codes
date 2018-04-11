#
# Util class for help training.
#

from math import sqrt

import numpy as np
import tensorflow as tf
from numpy import arange, power
from tensorflow import (Variable, matmul, reduce_mean, truncated_normal, zeros)

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

    def RELU_activation(self, activation, dropout_prob=None):
        """
        RELU layer.
        """
        relu_layer = tf.nn.relu(activation)
        if dropout_prob is not None:
            return tf.nn.dropout(relu_layer, dropout_prob)
        return relu_layer

    def loss_optimizer(self, y, activation, train_learning_rate, beta, weights):
        """
        Return loss and optimizer functions.
        """
        loss = reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=activation))

        reg = tf.nn.l2_loss(weights.pop())
        # Loss function using L2 Regularization
        for w in weights:
            reg += tf.nn.l2_loss(w)

        loss = reduce_mean(loss + beta * reg)
        optimizer = tf.train.GradientDescentOptimizer(
            train_learning_rate).minimize(loss)

        return loss, optimizer

    def create_hidden_layer(self, count_1_hide_layer, layer_position):
        """
        Helper method to create hidden-layer.

        Count of nodes on current layer depends on the count of previous layer.
        """
        former_count_hide_layer = int(count_1_hide_layer * power(0.5, layer_position - 1))
        current_count_hide_layer = int(count_1_hide_layer * power(0.5, layer_position))

        weights = Variable(truncated_normal(shape=[former_count_hide_layer, current_count_hide_layer],
                                            stddev=sqrt(2.0/former_count_hide_layer)))
        biases = Variable(zeros([current_count_hide_layer]))

        return weights, biases, current_count_hide_layer
