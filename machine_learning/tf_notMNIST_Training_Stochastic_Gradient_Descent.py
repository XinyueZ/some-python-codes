#
# Run NN, faster, multinomial logistic regression using stochastic gradient descent.
#
import numpy as np
import tensorflow as tf 
from tensorflow import Variable
from tensorflow import constant
from tensorflow import placeholder
from tensorflow import matmul
from tensorflow import reduce_mean
from tensorflow import global_variables_initializer
from training_helper import TrainingHelper

class TF_notMNIST_Training_Stochastic_Gradient_Descent:
        def __init__(self, each_object_size_width = 28, each_object_size_height = 28, train_batch = 1000, train_steps = 10000, train_learning_rate = 0.5):
            """
            Constructor.
            """
            self.each_object_size_width = each_object_size_width
            self.each_object_size_height = each_object_size_height
            self.train_batch = train_batch
            self.train_steps = train_steps
            self.train_learning_rate = train_learning_rate
            self.__accuracy__ = TrainingHelper().accuracy

        def __activation__(self, x, weights, biases):
            """
            Define math computation, the logits.
            """
            return tf.matmul(x, weights) + biases

        def __loss__optimizer__(self, y, activation):
            loss = reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits = activation))
            optimizer = tf.train.GradientDescentOptimizer(self.train_learning_rate).minimize(loss)
            return loss, optimizer

        def start_with(self, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels, count_classes, data_batch_size = 130):
            """
            Start multinomial logistic regression using simple gradient descent.
            """
            #
            # Changable values while training
            #
            tf_train_dataset = placeholder(tf.float32, shape = (data_batch_size, self.each_object_size_width * self.each_object_size_height))
            tf_train_labels = placeholder(tf.float32, shape = (data_batch_size, count_classes))
            #
            # Fixed values while training
            #
            tf_valid_dataset = constant(valid_dataset)
            tf_test_dataset = constant(test_dataset) 

            #
            # Variables should be trained
            # Classical weight and biases
            #
            tf_weight = Variable(tf.truncated_normal([self.each_object_size_width * self.each_object_size_height, count_classes]))
            tf_biases = Variable(tf.zeros([count_classes]))     

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
                    #
                    # TODO Can do more optimized batch computation.
                    #
                    offset = (step * data_batch_size) % (train_labels.shape[0] - data_batch_size)
                    batch_dataset = train_dataset[offset:(offset + data_batch_size), :]
                    batch_labels = train_labels[offset:(offset + data_batch_size), :]
 
                    # Per loop replace tf_train_dataset, tf_train_labels with batch.
                    _, ls, predications = sess.run(
                        [optimizer, loss, predication_for_train], 
                        feed_dict = {
                            tf_train_dataset: batch_dataset,
                            tf_train_labels: batch_labels
                        })
                    print("♻️ Batch with loss at step {}: {}, accuracy: {}%, validation accuracy: {}%"
                        .format(
                            step, 
                            ls, 
                            self.__accuracy__(predications, batch_labels),  
                            self.__accuracy__(predication_for_valid.eval(), valid_labels)
                        )
                        , sep=' ',  end = "\r", flush = True) 

                offset = (step * data_batch_size) % (train_labels.shape[0] - data_batch_size)
                batch_dataset = train_dataset[offset:(offset + data_batch_size), :]
                batch_labels = train_labels[offset:(offset + data_batch_size), :]
                _, ls, predications = sess.run(
                        [optimizer, loss, predication_for_train], 
                        feed_dict = {
                            tf_train_dataset: batch_dataset,
                            tf_train_labels: batch_labels
                        })
                print("👍 Final batch with loss at step {}: {}, accuracy: {}%, validation accuracy: {}%"
                        .format(
                            step, 
                            ls, 
                            self.__accuracy__(predications, batch_labels),  
                            self.__accuracy__(predication_for_valid.eval(), valid_labels)
                        )) 
                print('Test accuracy: %.1f%%' % self.__accuracy__(predication_for_test.eval(), test_labels))
