#
# Run NN, implementation of convolutional traning.
#
import config
import numpy as np
import tensorflow as tf
from tensorflow import (Variable, constant, global_variables_initializer, placeholder,
                        reduce_mean, truncated_normal, zeros)

from tf_training_helper import TrainingHelper


class TF_notMNIST_Training_Convolutional_Layer:
    def __init__(self, each_object_size_width=config.TRAIN_OBJECT_WIDTH, each_object_size_height=config.TRAIN_OBJECT_HEIGHT, train_steps=800, train_learning_rate=0.5, patch_size=5, channels=1, depth=16, hidden=64):
        """
        Constructor.
        """
        self.each_object_size_width = each_object_size_width
        self.each_object_size_height = each_object_size_height
        self.train_steps = train_steps
        self.train_learning_rate = train_learning_rate
        self.patch_size = patch_size
        self.channels = channels
        self.depth = depth
        self.hidden = hidden

        helper = TrainingHelper()
        self.__print_predications__ = helper.print_predications
        self.__print_test_accuracy__ = helper.print_test_accuracy
        self.__activation__ = helper.activation
        self.__loss_optimizer__ = helper.loss_optimizer
        self.__convolutional_model__ = helper.convolutional_model

    def start_with(self, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels, count_classes, data_batch_size=130):
        """
        Start multinomial logistic regression using simple gradient descent.
        """
        #
        # Changable values while training
        #
        tf_train_dataset = placeholder(tf.float32,
                                       shape=(data_batch_size, self.each_object_size_width, self.each_object_size_height, self.channels))
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
        tf_var_1_map = {
            "weights": Variable(truncated_normal(
                [self.patch_size, self.patch_size, self.channels, self.depth], stddev=0.1)),
            "biases": Variable(zeros([self.depth]))
        }
        tf_var_2_map = {
            "weights":  Variable(truncated_normal(
                [self.patch_size, self.patch_size,
                    self.depth, self.depth], stddev=0.1
            )),
            "biases": Variable(constant(1.0, shape=[self.depth]))
        }
        tf_var_3_map = {
            "weights": Variable(truncated_normal(
                [self.each_object_size_width // 4*self.each_object_size_height//4*self.depth, self.hidden], stddev=0.1
            )),
            "biases": Variable(constant(1.0, shape=[self.hidden]))
        }
        tf_var_4_map = {
            "weights": Variable(truncated_normal(
                [self.hidden, count_classes], stddev=0.1
            )),
            "biases":  Variable(constant(1.0, shape=[count_classes]))
        }

        #
        # Logits, loss and optimizer
        #
        logits = self.__convolutional_model__(
            tf_train_dataset,
            tf_var_1_map,
            tf_var_2_map,
            tf_var_3_map,
            tf_var_4_map
        )
        loss = reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf_train_labels, logits=logits)
        )
        optimizer = tf.train.GradientDescentOptimizer(
            self.train_learning_rate).minimize(loss)

        #
        # Convert dataset to predication
        # The actual problem is transformed into a probabilistic problem.
        #
        predication_for_train = tf.nn.softmax(logits)
        predication_for_valid = tf.nn.softmax(
            self.__convolutional_model__(
                tf_valid_dataset,
                tf_var_1_map,
                tf_var_2_map,
                tf_var_3_map,
                tf_var_4_map
            )
        )
        predication_for_test = tf.nn.softmax(
            self.__convolutional_model__(
                tf_test_dataset,
                tf_var_1_map,
                tf_var_2_map,
                tf_var_3_map,
                tf_var_4_map
            )
        )

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
                batch_labels = train_labels[offset: (
                    offset + data_batch_size), :]

                # Per loop replace tf_train_dataset, tf_train_labels with batch.
                _, ls, predications = sess.run(
                    [optimizer, loss, predication_for_train],
                    feed_dict={
                        tf_train_dataset: batch_dataset,
                        tf_train_labels: batch_labels
                    })
                self.__print_predications__(
                    step, ls, predications, batch_labels, predication_for_valid, valid_labels)

            self.__print_test_accuracy__(predication_for_test, test_labels)
