#
# Run NN, multinomial logistic regression using simple gradient descent.
#
import numpy as np
import tensorflow as tf
from tensorflow import (Variable, constant, global_variables_initializer,
                        matmul, reduce_mean, truncated_normal, zeros)

from training_helper import TrainingHelper


class TF_notMNIST_Training_Gradient_Descent:
    def __init__(self, each_object_size_width=28, each_object_size_height=28, train_batch=10000, train_steps=801, train_learning_rate=0.5):
        """
        Constructor.
        """
        self.each_object_size_width = each_object_size_width
        self.each_object_size_height = each_object_size_height
        self.train_batch = train_batch
        self.train_steps = train_steps
        self.train_learning_rate = train_learning_rate

        helper = TrainingHelper()
        self.__print_predications__ = helper.print_predications
        self.__print_test_accuracy__ = helper.print_test_accuracy
        self.__activation__ = helper.activation
        self.__loss_optimizer__ = helper.loss_optimizer

    def start_with(self, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels, count_classes, beta_for_regularizer=0.01):
        """
        Start multinomial logistic regression using simple gradient descent.
        """
        #
        # Fixed values while training
        #
        tf_train_dataset = constant(train_dataset[:self.train_batch, :])
        tf_train_labels = constant(train_labels[:self.train_batch])
        tf_valid_dataset = constant(valid_dataset)
        tf_test_dataset = constant(test_dataset)

        #
        # Variables should be trained.
        # Classical weight and biases.
        #
        tf_weights = Variable(truncated_normal(
            [self.each_object_size_width * self.each_object_size_height, count_classes]))
        tf_biases = Variable(zeros([count_classes]))

        logits = self.__activation__(tf_train_dataset, tf_weights, tf_biases)
        loss, optimizer = self.__loss_optimizer__(
            tf_train_labels, logits, self.train_learning_rate, beta_for_regularizer, [tf_weights])

        #
        # Convert dataset to predication
        # The actual problem is transformed into a probabilistic problem.
        #
        predication_for_train = tf.nn.softmax(logits)
        predication_for_valid = tf.nn.softmax(
            self.__activation__(tf_valid_dataset, tf_weights, tf_biases))
        predication_for_test = tf.nn.softmax(
            self.__activation__(tf_test_dataset, tf_weights, tf_biases))

        #
        # Training
        #
        print("\n")
        with tf.Session() as sess:
            init = global_variables_initializer()
            sess.run(init)
            for step in range(self.train_steps):
                _, ls, predications = sess.run(
                    [optimizer, loss, predication_for_train])
                self.__print_predications__(
                    step, ls, predications,  train_labels[:self.train_batch, :], predication_for_valid, valid_labels) 
            
            self.__print_test_accuracy__(predication_for_test, test_labels)
