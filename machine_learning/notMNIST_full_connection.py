import downloader
import extractor
import pickle_maker
import pickle_prune
from  tf_notMNIST_Training_Gradient_Descent import TF_notMNIST_Training_Gradient_Descent
from  tf_notMNIST_Training_Stochastic_Gradient_Descent import TF_notMNIST_Training_Stochastic_Gradient_Descent

CLASSES_TO_TRAIN = 10
TRAIN_BATCH = 1000
TRAIN_STEPS = 5000
TRAIN_LEARNING_RATE = 0.01

print("\n")
print("⛷ Traning: NN, multinomial logistic regression using simple gradient descent.")
print("\n")
 
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
print("\n")
 
TF_notMNIST_Training_Stochastic_Gradient_Descent(
        train_batch=TRAIN_BATCH, train_steps=TRAIN_STEPS, train_learning_rate=TRAIN_LEARNING_RATE
).start_with(
    pickle_prune.train_dataset, pickle_prune.train_labels, 
    pickle_prune.valid_dataset, pickle_prune.valid_labels, 
    pickle_prune.test_dataset, pickle_prune.test_labels, 
    CLASSES_TO_TRAIN
)
