#
# Util class for help training.
#

from math import sqrt

import config
import numpy as np
import tensorflow as tf
from numpy import (arange, argmax, sum)
from tensorflow import (Variable, matmul, reduce_mean,
                        truncated_normal, zeros, reshape)

from six.moves import cPickle as pickle


class TrainingHelper:
    def __init__(self, each_object_size_width=config.TRAIN_OBJECT_WIDTH, each_object_size_height=config.TRAIN_OBJECT_HEIGHT, channel=1):
        """
        Constructor 
        """
        self.each_object_size_width = each_object_size_width
        self.each_object_size_height = each_object_size_height
        self.channel = channel

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

    def flat_dataset_labels_with_channels(self, dataset, labels, count_classes):
        """
        Flat dataset, labels to 2-D arrays.
        """
        ds = dataset.reshape((-1,
                              self.each_object_size_width,
                              self.each_object_size_height,
                              self.channel)).astype(np.float32)
        lb = (arange(count_classes) == labels[:, None]).astype(np.float32)
        return ds, lb

    def flat_dataset_labels(self, dataset, labels, count_classes):
        """
        Flat dataset, labels to 2-D arrays.

        Consider this:

        list of image 2 X 3 in pixels
        z = np.array(
                [
                    [1,2,3],
                    [4,5,6]
                ],
                [
                    [7,8,9],
                    [10,11,12]
                ],
                [
                    [13,14,15],
                    [16,17,18]
        ])


        run: z.reshape((-1, 6)) to flat 2-D to 1-D array
        array([
            [ 1,  2,  3,  4,  5,  6],
            [ 7,  8,  9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18]])

        Consisder this:

        If: there're 2 types of image:

        y = np.array([0, #For 1. image type 
                      1, #For 2. image type
                      1])#For 3. image type
        run (np.arange(2)==y[:, None]).astype(np.float32)
        array([
            [1., 0.], # For 1. image
            [0., 1.], # For 2. image
            [0., 1.]],# For 3. image
        dtype=float32)

        If: there're 3 types of image:

        y = np.array([0, #For 1. image type 
                      1, #For 2. image type
                      2])#For 3. image type
        run (np.arange(3)==y[:, None]).astype(np.float32)
        array([
            [1., 0., 0.], # For 1. image
            [0., 1., 0.], # For 2. image
            [0., 0., 1.]],# For 3. image
        dtype=float32)
        """
        ds = dataset.reshape((-1, self.each_object_size_width *
                              self.each_object_size_height)).astype(np.float32)
        lb = (arange(count_classes) == labels[:, None]).astype(np.float32)
        return ds, lb

    def __accuracy__(self,  predictions, labels):
        return (100.0 * sum(argmax(predictions, 1) == argmax(labels, 1)) / predictions.shape[0])

    def activation(self, x, weights, biases):
        """
        Define math computation, the logits.
        """
        return matmul(x, weights) + biases

    def convolutional_model(self, data, layer_1, layer_2, layer_3, layer_4, padding_c="SAME"):
        c = tf.nn.conv2d(data,
                         layer_1["weights"],
                         [1, 2, 2, 1],
                         padding=padding_c)
        hidden = tf.nn.relu(c + layer_1["biases"])

        c = tf.nn.conv2d(hidden,
                         layer_2["weights"],
                         [1, 2, 2, 1],
                         padding=padding_c)
        hidden = tf.nn.relu(c+layer_2["biases"])

        shape = hidden.get_shape().as_list()
        _reshape_ = reshape(hidden, [shape[0], shape[1]*shape[2]*shape[3]])
        hidden = tf.nn.relu(
            matmul(_reshape_, layer_3["weights"]) + layer_3["biases"])

        return matmul(hidden, layer_4["weights"]) + layer_4["biases"]

    def RELU_activation(self, activation, dropout_prob=None):
        """
        RELU layer.
        """
        relu_layer = tf.nn.relu(activation)
        if dropout_prob is not None:
            return tf.nn.dropout(relu_layer, dropout_prob)
        return relu_layer

    def loss_optimizer(self, y, activation, train_learning_rate, beta, weights, train_steps=None):
        """
        Return loss and optimizer functions.
        """
        loss = reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=activation))

        holder = weights.pop()
        reg = tf.nn.l2_loss(holder)
        # Loss function using L2 Regularization
        for w in weights:
            reg += tf.nn.l2_loss(w)
        weights.append(holder)

        loss = reduce_mean(loss + beta * reg)
        optimizer = None
        if train_steps is None:
            optimizer = tf.train.GradientDescentOptimizer(
                train_learning_rate).minimize(loss)
        else:
            optimizer = tf.train.GradientDescentOptimizer(
                train_learning_rate).minimize(loss, global_step=train_steps)
        return loss, optimizer

    def create_hidden_layer(self, former_count_hide_layer, current_count_hide_layer):
        """
        Helper method to create hidden-layer.

        Count of nodes on current layer depends on the count of previous layer.
        """
        weights = Variable(truncated_normal(shape=[former_count_hide_layer, current_count_hide_layer],
                                            stddev=sqrt(2.0/former_count_hide_layer)))
        biases = Variable(zeros([current_count_hide_layer]))
        return weights, biases

    def create_exponential_rate(self, start_learning_rate, training_train_steps):
        return tf.train.exponential_decay(start_learning_rate, training_train_steps, 100000, 0.96, staircase=True)

    def print_test_accuracy(self, predication_for_test, test_labels):
        print('👍 Test accuracy: {:.2f}'.format(self.__accuracy__(
            predication_for_test.eval(), test_labels)))

    def print_predications(self, step, loss, predications, batch_labels, predication_for_valid, valid_labels):
        print("♻️ Batch with loss at step {}: {:2.4f}, accuracy: {:.2f}, validation accuracy: {:.2f}"
              .format(
                  step,
                  loss,
                  self.__accuracy__(predications, batch_labels),
                  self.__accuracy__(
                      predication_for_valid.eval(), valid_labels)
              ))
