#
# Run NN, faster, multinomial logistic regression using stochastic gradient descent.
#
import numpy as np
import tensorflow as tf
from tensorflow import (Variable, constant, global_variables_initializer,
                        matmul, placeholder, reduce_mean)

from training_helper import TrainingHelper


class TF_notMNIST_Training_Stochastic_Gradient_Descent:
    def __init__(self, each_object_size_width=28, each_object_size_height=28,  train_steps=10000, train_learning_rate=0.5):
        """
        Constructor.
        """
        self.each_object_size_width = each_object_size_width
        self.each_object_size_height = each_object_size_height
        self.train_steps = train_steps
        self.train_learning_rate = train_learning_rate

        helper = TrainingHelper()
        self.__accuracy__ = helper.accuracy
        self.__activation__ = helper.activation
        self.__loss_optimizer__ = helper.loss_optimizer

    def start_with(self, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels, count_classes, data_batch_size=130, beta_for_regularizer=0.01):
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
        tf_weights = Variable(tf.truncated_normal(
            [self.each_object_size_width * self.each_object_size_height, count_classes]))
        tf_biases = Variable(tf.zeros([count_classes]))

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
                        tf_train_labels: batch_labels
                    })
                print("♻️ Batch with loss at step {}: {:2.4f}, accuracy: {:.2f}, validation accuracy: {:.2f}"
                      .format(
                          step,
                          ls,
                          self.__accuracy__(predications, batch_labels),
                          self.__accuracy__(
                              predication_for_valid.eval(), valid_labels)
                      ), sep=' ',  end="\r", flush=True)

            offset = (
                step * data_batch_size) % (train_labels.shape[0] - data_batch_size)
            batch_dataset = train_dataset[offset:(offset + data_batch_size), :]
            batch_labels = train_labels[offset:(offset + data_batch_size), :]
            _, ls, predications = sess.run(
                [optimizer, loss, predication_for_train],
                feed_dict={
                    tf_train_dataset: batch_dataset,
                    tf_train_labels: batch_labels
                })
            print("👍 Final batch with loss at step {}: {:2.4f}, accuracy: {:.2f}, validation accuracy: {:.2f}"
                  .format(
                      step,
                      ls,
                      self.__accuracy__(predications, batch_labels),
                      self.__accuracy__(
                          predication_for_valid.eval(), valid_labels)
                  ))
            print('Test accuracy: {:.2f}'.format(self.__accuracy__(
                predication_for_test.eval(), test_labels)))
