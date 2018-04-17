import os

os.system("clear")

import config
import downloader
import extractor
import pickle_maker
import pickle_prune
from tf_notMNIST_Training_Gradient_Descent import \
    TF_notMNIST_Training_Gradient_Descent
from tf_notMNIST_Training_Multi_Relu_Layer_Gradient_Descent import \
    TF_notMNIST_Training_Multi_RELU_Layer_Stochastic_Gradient_Descent
from tf_notMNIST_Training_Relu_Layer_Gradient_Descent import \
    TF_notMNIST_Training_RELU_Layer_Stochastic_Gradient_Descent
from tf_notMNIST_Training_Stochastic_Gradient_Descent import \
    TF_notMNIST_Training_Stochastic_Gradient_Descent
from tf_notMNIST_Training_Convolutional_Layer import TF_notMNIST_Training_Convolutional_Layer
from tf_training_helper import TrainingHelper
from tf_notMNIST_Training_Premade_Estimator import TF_notMNIST_Training_Premade_Estimator

TRAIN_BATCH = 10000
TRAIN_STEPS = 1500
TRAIN_LEARNING_RATE = 0.5
HIDE_LAYER = 1024  # Nodes on hidden-layout
TOTAL_HIDDEN_LAYERS = 5  # How many hidden-layers.

training_helper = TrainingHelper()

print("► reformat total.pickle for Premade Estimator.")
train_dataset, train_labels = training_helper.flat_dataset_labels(
    pickle_prune.train_dataset,
    pickle_prune.train_labels,
    config.CLASSES_TO_TRAIN)
valid_dataset, valid_labels = training_helper.flat_dataset_labels(
    pickle_prune.valid_dataset,
    pickle_prune.valid_labels,
    config.CLASSES_TO_TRAIN)
test_dataset, test_labels = training_helper.flat_dataset_labels(
    pickle_prune.test_dataset,
    pickle_prune.test_labels,
    config.CLASSES_TO_TRAIN)

TF_notMNIST_Training_Premade_Estimator(TRAIN_STEPS).start_with(train_dataset,
                                                        pickle_prune.train_labels, # Don't use one-hot, https://codeburst.io/use-tensorflow-dnnclassifier-estimator-to-classify-mnist-dataset-a7222bf9f940
                                                        test_dataset,
                                                        pickle_prune.test_labels,
                                                        HIDE_LAYER,
                                                        TOTAL_HIDDEN_LAYERS,
                                                        config.CLASSES_TO_TRAIN)


print("► reformat total.pickle for Convolutional.")
train_dataset, train_labels = training_helper.flat_dataset_labels_with_channels(
    pickle_prune.train_dataset,
    pickle_prune.train_labels,
    config.CLASSES_TO_TRAIN)
valid_dataset, valid_labels = training_helper.flat_dataset_labels_with_channels(
    pickle_prune.valid_dataset,
    pickle_prune.valid_labels,
    config.CLASSES_TO_TRAIN)
test_dataset, test_labels = training_helper.flat_dataset_labels_with_channels(
    pickle_prune.test_dataset,
    pickle_prune.test_labels,
    config.CLASSES_TO_TRAIN)

print("👍 ")
print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


print("\n")
print("⛷ Traning: NN with Convolutional Model.")

TF_notMNIST_Training_Convolutional_Layer(
    train_steps=TRAIN_STEPS, train_learning_rate=TRAIN_LEARNING_RATE/10
).start_with(
    train_dataset, train_labels,
    valid_dataset, valid_labels,
    test_dataset, test_labels,
    config.CLASSES_TO_TRAIN
)

print("► reformat total.pickle for Linear.")
train_dataset, train_labels = training_helper.flat_dataset_labels(
    pickle_prune.train_dataset,
    pickle_prune.train_labels,
    config.CLASSES_TO_TRAIN)
valid_dataset, valid_labels = training_helper.flat_dataset_labels(
    pickle_prune.valid_dataset,
    pickle_prune.valid_labels,
    config.CLASSES_TO_TRAIN)
test_dataset, test_labels = training_helper.flat_dataset_labels(
    pickle_prune.test_dataset,
    pickle_prune.test_labels,
    config.CLASSES_TO_TRAIN)

print("👍 ")
print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

print("\n")
print("⛷ Traning: NN, fast and quckly, the stochastic gradient descent training with multiple RELU layers.")

TF_notMNIST_Training_Multi_RELU_Layer_Stochastic_Gradient_Descent(
    train_steps=TRAIN_STEPS, train_learning_rate=TRAIN_LEARNING_RATE
).start_with(
    train_dataset, train_labels,
    valid_dataset, valid_labels,
    test_dataset, test_labels,
    # For this training, it is used in first hidden-layer for second hidden-layer.
    HIDE_LAYER,
    TOTAL_HIDDEN_LAYERS,
    config.CLASSES_TO_TRAIN
)


print("\n")
print("⛷ Traning: NN, fast and quckly, the stochastic gradient descent training with one RELU layer.")

TF_notMNIST_Training_RELU_Layer_Stochastic_Gradient_Descent(
    train_steps=TRAIN_STEPS, train_learning_rate=TRAIN_LEARNING_RATE
).start_with(
    # Avoide overfitting
    train_dataset[:500, :], train_labels[:500],
    valid_dataset, valid_labels,
    test_dataset, test_labels,
    HIDE_LAYER,
    config.CLASSES_TO_TRAIN
)


print("\n")
print("⛷ Traning: NN, fast and quckly, the stochastic gradient descent training.")


TF_notMNIST_Training_Stochastic_Gradient_Descent(
    train_steps=TRAIN_STEPS, train_learning_rate=TRAIN_LEARNING_RATE
).start_with(
    train_dataset, train_labels,
    valid_dataset, valid_labels,
    test_dataset, test_labels,
    config.CLASSES_TO_TRAIN
)


print("\n")
print("⛷ Traning: NN, multinomial logistic regression using simple gradient descent.")


TF_notMNIST_Training_Gradient_Descent(
    train_batch=TRAIN_BATCH, train_steps=TRAIN_STEPS, train_learning_rate=TRAIN_LEARNING_RATE
).start_with(
    train_dataset, train_labels,
    valid_dataset, valid_labels,
    test_dataset, test_labels,
    config.CLASSES_TO_TRAIN
)