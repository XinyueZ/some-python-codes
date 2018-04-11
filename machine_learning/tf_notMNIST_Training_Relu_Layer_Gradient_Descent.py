#
# Run NN, faster, multinomial logistic regression using stochastic gradient descent
# with one RELU layer.
#
import numpy as np
import tensorflow as tf
from tensorflow import (Variable, constant, global_variables_initializer,
                        matmul, placeholder, reduce_mean, zeros)

from training_helper import TrainingHelper


class TF_notMNIST_Training_RELU_Layer_Stochastic_Gradient_Descent :
    def __init__(self, each_object_size_width=28, each_object_size_height=28,  train_steps=10000, train_learning_rate=0.5):
        """
        Constructor.
        """
        self.each_object_size_width = each_object_size_width
        self.each_object_size_height = each_object_size_height
        self.train_steps = train_steps
        self.train_learning_rate = train_learning_rate

        helper = TrainingHelper()
        self.__print_predications__ = helper.print_predications
        self.__print_test_accuracy__ = helper.print_test_accuracy
        self.__activation__ = helper.activation
        self.__loss_optimizer__ = helper.loss_optimizer
        self.__RELU_activation__ = helper.RELU_activation

    def start_with(self, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels, count_hide_layer, count_classes, data_batch_size=130, beta_for_regularizer=0.01, dropout_prob=0.5):
        """
        Start multinomial logistic regression using simple gradient descent.
        """
        #
        # Changable values while training
        #
        tf_train_dataset = placeholder(tf.float32, shape=(
            data_batch_size, self.each_object_size_width * self.each_object_size_height))
        tf_train_labels = placeholder(
            tf.float32, shape=(data_batch_size, count_classes))
        #
        # Fixed values while training
        #
        tf_valid_dataset = constant(valid_dataset)
        tf_test_dataset = constant(test_dataset)

        #
        # Variables should be trained.
        # Classical weight and biases.
        #
        tf_weights_1 = Variable(tf.truncated_normal(
            [self.each_object_size_width * self.each_object_size_height, count_hide_layer]))
        tf_biases_1 = Variable(zeros([count_hide_layer]))

        tf_weights_2 = Variable(tf.truncated_normal(
            [count_hide_layer, count_classes]))
        tf_biases_2 = Variable(zeros([count_classes]))

        # Hidden-layer-1
        tf_dropout_prob = placeholder(tf.float32)
        logits_1 = self.__RELU_activation__(
            self.__activation__(tf_train_dataset, tf_weights_1, tf_biases_1),
            tf_dropout_prob)

        # Output-layer which connects with logits_1.
        logits_2 = self.__activation__(logits_1, tf_weights_2, tf_biases_2)

        # Loss and optimizer
        loss, optimizer = self.__loss_optimizer__(
            tf_train_labels, logits_2, self.train_learning_rate, beta_for_regularizer, [tf_weights_1, tf_weights_2])

        #
        # Convert dataset to predication
        # The actual problem is transformed into a probabilistic problem.
        #
        # Softmax the last layer always.
        predication_for_train = tf.nn.softmax(logits_2)

        predication_for_valid = tf.nn.softmax(
            self.__activation__(
                self.__RELU_activation__(
                    self.__activation__(tf_valid_dataset, tf_weights_1, tf_biases_1)), tf_weights_2, tf_biases_2))

        predication_for_test = tf.nn.softmax(
            self.__activation__(
                self.__RELU_activation__(
                    self.__activation__(tf_test_dataset, tf_weights_1, tf_biases_1)), tf_weights_2, tf_biases_2))

        #
        # Training
        #
        print("\n")
        with tf.Session() as sess:
            init = global_variables_initializer()
            sess.run(init)
            for step in range(self.train_steps):
                #
                # TODO Can do more optimized batch computation.
                #
                offset = (
                    step * data_batch_size) % (train_labels.shape[0] - data_batch_size)
                batch_dataset = train_dataset[offset:(
                    offset + data_batch_size), :]
                batch_labels = train_labels[offset:(
                    offset + data_batch_size), :]

                # Per loop replace tf_train_dataset, tf_train_labels with batch.
                _, ls, predications = sess.run(
                    [optimizer, loss, predication_for_train],
                    feed_dict={
                        tf_train_dataset: batch_dataset,
                        tf_train_labels: batch_labels,
                        tf_dropout_prob: dropout_prob
                    })
                self.__print_predications__(step, ls, predications, batch_labels, predication_for_valid, valid_labels)
 
            self.__print_test_accuracy__(predication_for_test, test_labels) 
