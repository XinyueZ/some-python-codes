#
# Run NN, multinomial logistic regression using simple gradient descent.
#
import numpy as np
import tensorflow as tf
from tensorflow import (Variable, constant, global_variables_initializer,
                        matmul, reduce_mean, truncated_normal, zeros)

from training_helper import TrainingHelper


class TF_notMNIST_Training_Gradient_Descent:
        def __init__(self, each_object_size_width = 28, each_object_size_height = 28, train_batch = 1000, train_steps = 801, train_learning_rate = 0.5):
            """
            Constructor.
            """
            self.each_object_size_width = each_object_size_width
            self.each_object_size_height = each_object_size_height
            self.train_batch = train_batch
            self.train_steps = train_steps
            self.train_learning_rate = train_learning_rate
            self.__accuracy__ = TrainingHelper().accuracy

        def __activation__(self, tf_dataset, weights, biases):
            """
            Define math computation, the logits.
            """
            return matmul(tf_dataset, weights) + biases

        def __loss__optimizer__(self, tf_train_labels, activation):
            loss = reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits = activation))
            optimizer = tf.train.GradientDescentOptimizer(self.train_learning_rate).minimize(loss)
            return loss, optimizer

        def start_with(self, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels, count_classes):
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
            # Variables should be trained
            # Classical weight and biases
            #
            tf_weight = Variable(truncated_normal([self.each_object_size_width * self.each_object_size_height, count_classes]))
            tf_biases = Variable(zeros([count_classes]))     

            logits = self.__activation__(tf_train_dataset, tf_weight, tf_biases)
            loss, optimizer = self.__loss__optimizer__(tf_train_labels, logits)

            # 
            # Convert dataset to predication
            # The actual problem is transformed into a probabilistic problem.
            #
            predication_for_train = tf.nn.softmax(logits)
            predication_for_valid = tf.nn.softmax(self.__activation__(tf_valid_dataset, tf_weight, tf_biases))
            predication_for_test = tf.nn.softmax(self.__activation__(tf_test_dataset, tf_weight, tf_biases))

            #
            # Training
            #
            print("\n")
            with tf.Session() as sess:
                init = global_variables_initializer()
                sess.run(init)
                for step in range(self.train_steps):
                    _, ls, predications = sess.run([optimizer, loss, predication_for_train])
                    print("♻️ Loss at step {}: {}, Training accuracy: {}%, Validation accuracy: {}%"
                        .format(
                            step, 
                            ls, 
                            self.__accuracy__(predications, train_labels[:self.train_batch, :]),  
                            self.__accuracy__(predication_for_valid.eval(), valid_labels)
                        )
                        , sep=' ',  end = "\r", flush = True) 

                _, ls, predications = sess.run([optimizer, loss, predication_for_train])
                print("👍 Final loss at step {}: {}, Training accuracy: {}%, Validation accuracy: {}%"
                        .format(
                            step, 
                            ls, 
                            self.__accuracy__(predications, train_labels[:self.train_batch, :]),  
                            self.__accuracy__(predication_for_valid.eval(), valid_labels)
                        )) 
                print('Test accuracy: %.1f%%' % self.__accuracy__(predication_for_test.eval(), test_labels))
