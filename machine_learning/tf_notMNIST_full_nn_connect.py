import os
os.system("clear")

import downloader
import extractor
import pickle_maker
import pickle_prune
from tf_notMNIST_Training_Gradient_Descent import \
    TF_notMNIST_Training_Gradient_Descent
from tf_notMNIST_Training_Stochastic_Gradient_Descent import \
    TF_notMNIST_Training_Stochastic_Gradient_Descent
from tf_notMNIST_Training_Relu_Layer_Gradient_Descent import \
    TF_notMNIST_Training_RELU_Layer_Stochastic_Gradient_Descent


CLASSES_TO_TRAIN = 10
TRAIN_BATCH = 10000
TRAIN_STEPS = 500
TRAIN_LEARNING_RATE = 0.01
HIDE_LAYER = 2048


print("\n")
print("⛷ Traning: NN, multinomial logistic regression using simple gradient descent.")


TF_notMNIST_Training_Gradient_Descent(
    train_batch=TRAIN_BATCH, train_steps=TRAIN_STEPS, train_learning_rate=TRAIN_LEARNING_RATE
).start_with(
    pickle_prune.train_dataset, pickle_prune.train_labels,
    pickle_prune.valid_dataset, pickle_prune.valid_labels,
    pickle_prune.test_dataset, pickle_prune.test_labels,
    CLASSES_TO_TRAIN
)

print("\n")
print("⛷ Traning: NN, fast and quckly, the stochastic gradient descent training.")


TF_notMNIST_Training_Stochastic_Gradient_Descent(
    train_steps=TRAIN_STEPS, train_learning_rate=TRAIN_LEARNING_RATE
).start_with(
    pickle_prune.train_dataset, pickle_prune.train_labels,
    pickle_prune.valid_dataset, pickle_prune.valid_labels,
    pickle_prune.test_dataset, pickle_prune.test_labels,
    CLASSES_TO_TRAIN
)


print("\n")
print("⛷ Traning: NN, fast and quckly, the stochastic gradient descent training with RELU layer.")

TF_notMNIST_Training_RELU_Layer_Stochastic_Gradient_Descent(
    train_steps=TRAIN_STEPS, train_learning_rate=TRAIN_LEARNING_RATE
).start_with(
    # Avoide overfitting
    pickle_prune.train_dataset[:500, :], pickle_prune.train_labels[:500],
    pickle_prune.valid_dataset, pickle_prune.valid_labels,
    pickle_prune.test_dataset, pickle_prune.test_labels,
    HIDE_LAYER,
    CLASSES_TO_TRAIN
)