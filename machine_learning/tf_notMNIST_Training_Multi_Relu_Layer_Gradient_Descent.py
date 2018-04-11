#
# Run NN, faster, multinomial logistic regression using stochastic gradient descent
# with multiple RELU layers.
#
import numpy as np
import tensorflow as tf
from tensorflow import (Variable, constant, global_variables_initializer,
                        matmul, placeholder, reduce_mean, zeros, truncated_normal)

from training_helper import TrainingHelper


class TF_notMNIST_Training_Multi_RELU_Layer_Stochastic_Gradient_Descent:
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
        self.__RELU_activation__ = helper.RELU_activation
        self.__create_hidden_layer__ = helper.create_hidden_layer
        self.__create_exponential_rate__ = helper.create_exponential_rate

    def start_with(self, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels, count_hide_layer, total_hidden_layers, count_classes, data_batch_size=130, beta_for_regularizer=0.01, dropout_prob=0.5):
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
        # More weights and biases.
        #
        tf_weights_list = []
        tf_biases_list = []
        count_1_hide_layer = self.each_object_size_width * self.each_object_size_height
        former_count_hide_layer = count_1_hide_layer
        layer_index = 0
        while(layer_index < total_hidden_layers):
            tf_weights, tf_biases, former_count_hide_layer = self.__create_hidden_layer__(
                count_1_hide_layer, layer_index + 1)
            tf_weights_list.append(tf_weights)
            tf_biases_list.append(tf_biases)
            layer_index += 1

        tf_weights_output = Variable(tf.truncated_normal(
            [former_count_hide_layer, count_classes]))
        tf_biases_output = Variable(zeros([count_classes]))
        tf_weights_list.append(tf_weights_output)
        tf_biases_list.append(tf_biases_output)

        #
        # Define hidden-layers.
        #
        tf_dropout_prob = placeholder(tf.float32)
        logits_last_hidden_layout = self.__RELU_activation__(
            self.__activation__(
                tf_train_dataset, tf_weights_list[0], tf_biases_list[0]),
            tf_dropout_prob)
        layer_index = 1
        while(layer_index < total_hidden_layers):
            logits_last_hidden_layout = self.__RELU_activation__(
                self.__activation__(
                    logits_last_hidden_layout, tf_weights_list[layer_index], tf_biases_list[layer_index]),
                tf_dropout_prob)
            layer_index += 1

        # Output-layer which connects with logits of last hidden-layer.
        logits_output = self.__activation__(
            logits_last_hidden_layout, tf_weights_output, tf_biases_output)

        #
        # Loss and optimizer
        #
        loss, optimizer = self.__loss_optimizer__(
            tf_train_labels,
            logits_output,
            self.__create_exponential_rate__(
                self.train_learning_rate, self.train_steps),
            beta_for_regularizer,
            tf_weights_list)

        #
        # Convert dataset to predication
        # The actual problem is transformed into a probabilistic problem.
        #
        # Softmax the last layer always.
        #

        #
        # predication for train
        #
        predication_for_train = tf.nn.softmax(logits_output)

        #
        # predication for varlidation
        #
        last_predication_for_valid = self.__RELU_activation__(
            self.__activation__(tf_valid_dataset, tf_weights_list[0], tf_biases_list[0]))
        layer_index = 1
        while layer_index < total_hidden_layers:
            last_predication_for_valid = self.__RELU_activation__(
                self.__activation__(last_predication_for_valid, tf_weights_list[layer_index], tf_biases_list[layer_index]))
            layer_index += 1
        predication_for_valid = tf.nn.softmax(
            self.__activation__(last_predication_for_valid, tf_weights_output, tf_biases_output))

        #
        # predication for test
        #
        last_predication_for_test_dataset = self.__RELU_activation__(
            self.__activation__(tf_test_dataset, tf_weights_list[0], tf_biases_list[0]))
        layer_index = 1
        while layer_index < total_hidden_layers:
            last_predication_for_test_dataset = self.__RELU_activation__(
                self.__activation__(last_predication_for_test_dataset, tf_weights_list[layer_index], tf_biases_list[layer_index]))
            layer_index += 1
        predication_for_test = tf.nn.softmax(
            self.__activation__(last_predication_for_test_dataset, tf_weights_output, tf_biases_output))

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
                        tf_train_labels: batch_labels,
                        tf_dropout_prob: dropout_prob
                    })
                print("♻️ Batch with loss at step {}: {:2.4f}, accuracy: {:.1f}%, validation accuracy: {:.1f}%"
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
                    tf_train_labels: batch_labels,
                    tf_dropout_prob: dropout_prob
                })
            print("👍 Final batch with loss at step {}: {:2.4f}, accuracy: {:.1f}%, validation accuracy: {:.1f}%"
                  .format(
                      step,
                      ls,
                      self.__accuracy__(predications, batch_labels),
                      self.__accuracy__(
                          predication_for_valid.eval(), valid_labels)
                  ))
            print('Test accuracy: {:.1f}%'.format(self.__accuracy__(
                predication_for_test.eval(), test_labels)))
