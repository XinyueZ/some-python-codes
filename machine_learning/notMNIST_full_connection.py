import downloader
import extractor
import pickle_maker
import pickle_prune
from  tf_notMNIST import TF_notMNIST_Training_Gradient_Descent

tf_notMNIST_Training = TF_notMNIST_Training_Gradient_Descent()
tf_notMNIST_Training.start_with(
    pickle_prune.train_dataset, pickle_prune.train_labels, 
    pickle_prune.valid_dataset, pickle_prune.valid_labels, 
    pickle_prune.test_dataset, pickle_prune.test_labels, 
    10
)

