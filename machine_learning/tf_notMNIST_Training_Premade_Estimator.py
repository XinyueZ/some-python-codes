#
# Run NN using Premade Estimator.
#
import config
import numpy as np
import tensorflow as tf
from tensorflow import (feature_column)

from tf_training_helper import TrainingHelper


class TF_notMNIST_Training_Premade_Estimator:
    def __init__(self, train_steps=10000):
        """
        Constructor.
        """
        self.train_steps = train_steps

    def __train_input_fn__(self, features, labels, batch_size):
        """An input function for training"""
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        # Shuffle, repeat, and batch the examples.
        return dataset.shuffle(1000).repeat().batch(batch_size)

    def __eval_input_fn__(self, features, labels=None, batch_size=None):
        """An input function for evaluation or prediction"""
        if labels is None:
            # No labels, use only features.
            inputs = features
        else:
            inputs = (features, labels)

        # Convert inputs to a tf.dataset object.
        dataset = tf.data.Dataset.from_tensor_slices(inputs)

        # Batch the examples
        assert batch_size is not None, "batch_size must not be None"
        dataset = dataset.batch(batch_size)

        # Return the read end of the pipeline.
        return dataset.make_one_shot_iterator().get_next()

    def start_with(self, train_dataset, train_labels,  test_dataset, test_labels, count_hide_layer, total_hidden_layers, count_classes, data_batch_size=130):
        """
        Start multinomial logistic regression using simple gradient descent.
        """
        tf_train_data = dict(enumerate(np.rot90(train_dataset)))
        tf_train_data = {str(k): v for k, v in tf_train_data.items()}
        tf_train_labels = train_labels

        tf_test_data = dict(enumerate(np.rot90(test_dataset)))
        tf_test_data = {str(k): v for k, v in tf_test_data.items()}
        tf_test_labels = test_labels

        feature_columns = []
        for key in tf_train_data.keys():
            feature_columns.append(tf.feature_column.numeric_column(key=key))

        # Train the model.
        hidden_units = []
        for _ in range(total_hidden_layers):
            hidden_units.append(count_hide_layer)

        classifier = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=hidden_units,
            n_classes=count_classes)

        print("♻️ training with DNNClassifier...")
        classifier.train(
            input_fn=lambda: self.__train_input_fn__(
                tf_train_data,
                tf_train_labels,
                data_batch_size),
            steps=self.train_steps)

        # Predict test-data.
        print("♻️ predict ...")
        predictions = classifier.predict(
            input_fn=lambda: self.__eval_input_fn__(
                tf_test_data,
                None,
                data_batch_size))

        
        for pred_dict in predictions:
            template = ("class type:{:<10}{:^}{:>5.2f}%")
            class_id = pred_dict['class_ids'][0]
            probability = pred_dict['probabilities'][class_id]
            print(template.format(
                class_id,
                "->",
                100 * probability))
        print("👍")

        print("♻️ evaluate ...")
        eval_result = classifier.evaluate(
            input_fn=lambda: self.__eval_input_fn__(tf_test_data, tf_test_labels, data_batch_size))

        print('👍 Test set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
