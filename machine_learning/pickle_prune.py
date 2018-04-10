#
# Util class for prune pickle results.
#

import numpy as np
from numpy import ndarray 
from numpy import arange
from numpy.random import shuffle as shuffle_all
from numpy.random import permutation as shuffle_all_and_copy
from os.path import isdir as is_dir
from os.path import splitext as split_text
from os import listdir as directory_list
from os.path import join as path_join
from os import stat as stat_info
from six.moves import cPickle as pickle


class PicklePrune:
    def __init__(self, pickle_fullname_list, train_size, valid_size = 0, each_object_size_width = 28, each_object_size_height = 28):
        """
        Merge and prune the training data as needed. Depending on your computer setup, you might not be able to fit it all in memory, and you can tune train_size as needed. The labels will be stored into a separate array of integers 0 through 9. Also create a validation dataset for hyperparameter tuning.
        """
        self.pickle_fullname_list = pickle_fullname_list
        self.train_size = train_size
        self.valid_size = valid_size
        self.each_object_size_width = each_object_size_width
        self.each_object_size_height = each_object_size_height

    def __create_dataset_labels__(self, rows):
        """
        Create array with 3-D, ndarray with shape: (rows, self.each_object_size_width, self.each_object_size_height).
        Create array with 1-D, ndarray with shape: (rows).

        return arrays if rows >= 0 otherwise all Nones.
        """
        if rows:
            return  ndarray((rows,  self.each_object_size_width, self.each_object_size_height), dtype=np.float32), ndarray(rows, dtype=np.int32)
        else:
            return None, None

    def __randomize__(self, dataset, labels):
        """
        Shuffles dataset and lables and returns copy of them.
        """
        permutation = shuffle_all_and_copy(labels.shape[0])
        return dataset[permutation, :, :], labels[permutation]

    def prune(self, randomize = False):
        """
        Prune ".pickle" to memory. It will revert data in .pickle(see pickle.maker.py).
        """
        # Here is how many objects will be predicated in the furture.
        count_classes = len(self.pickle_fullname_list)

        # Get train's dataset and labels, it might be None, None when the self.train_size is 0.
        train_dataset, train_labels = self.__create_dataset_labels__(self.train_size)

        # Get valid's dataset and labels, it might be None, None when the self.valid_size is 0.
        valid_dataset, valid_labels = self.__create_dataset_labels__(self.valid_size)

        # Define batch. It defines how many items will be used per loop, see blow.
        train_batch_each_class = self.train_size // count_classes
        valid_batch_each_class = self.valid_size // count_classes

        # Prepare array operations with different indecs.
        # All start with 0, 0, end with batch.
        start_train, start_valid = 0, 0
        end_train, end_valid = train_batch_each_class, valid_batch_each_class

        end_train_plus_valid = train_batch_each_class + valid_batch_each_class

        for label, pickle_file in enumerate(self.pickle_fullname_list):
            print("► label:{}".format(label), sep=' ',  end = "\r", flush = True)
            try:
                with open(pickle_file, "rb") as in_file:
                    objects_set = pickle.load(in_file)
                    shuffle_all(objects_set)

                    train_dataset[start_train:end_train, :, :] = objects_set[valid_batch_each_class:end_train_plus_valid, :, :] # TODO Need discuss ...
                    train_labels[start_train:end_train] = label
                    start_train += train_batch_each_class
                    end_train += train_batch_each_class

                    if valid_dataset is not None: # Notice: If self.valid_size is 0, here is None for valid_dataset.
                        valid_dataset[start_valid:end_valid, :, :] = objects_set[:valid_batch_each_class, :, :]
                        valid_labels[start_valid:end_valid] = label
                        # start position moves with gap of batch. 0 ->
                        # end position moves with gap of batch. batch ->
                        start_valid += valid_batch_each_class
                        end_valid += valid_batch_each_class
            except Exception as e:
                print('Unable to read {} : {}'.format(pickle_file,  e))
                raise

        if randomize:
            train_dataset, train_labels = self.__randomize__(train_dataset, train_labels)
            if valid_dataset is not None:
                valid_dataset, valid_labels = self.__randomize__(valid_dataset, valid_labels)

        return train_dataset, train_labels, valid_dataset, valid_labels
    

"""
Make a summary pickle to total.pickle
"""
def get_prune_pickles(src_root):
    """
    Return list of "xxx.pickle" under src_root.
    """
    li = directory_list(src_root)
    input_objects = []

    for input_data_name in li:
        full_name = path_join(src_root, input_data_name)
        if not is_dir(full_name):
            extension = split_text(input_data_name)[1]
            if extension == ".pickle":
                input_objects.append(full_name)

    return input_objects



print("► try classes with pickles.")

prune_pickles = get_prune_pickles("./notMNIST_large")
pickle_prune = PicklePrune(prune_pickles, 200000, 10000)
train_dataset, train_labels, valid_dataset, valid_labels =  pickle_prune.prune()

prune_pickles = get_prune_pickles("./notMNIST_small")
pickle_prune = PicklePrune(prune_pickles, 10000)
test_dataset, test_labels, _, _  =  pickle_prune.prune()

print("👍 ")
print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

print("► randomize.")
prune_pickles = get_prune_pickles("./notMNIST_large")
pickle_prune = PicklePrune(prune_pickles, 200000, 10000)
train_dataset, train_labels, valid_dataset, valid_labels =  pickle_prune.prune(True)

prune_pickles = get_prune_pickles("./notMNIST_small")
pickle_prune = PicklePrune(prune_pickles, 10000)
test_dataset, test_labels, _, _  =  pickle_prune.prune(True)

print("👍 ")
print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


from  training_helper import TrainingHelper

print("► save total.pickle.")
training_helper = TrainingHelper()
save_pickle = path_join(".", "totals.pickle")
data_to_save = {
        "train_dataset": train_dataset,
        "train_labels": train_labels,
        "valid_dataset": valid_dataset,
        "valid_labels": valid_labels,
        "test_dataset": test_dataset,
        "test_labels": test_labels
    }
training_helper.save_pickle(save_pickle, data_to_save)
info = stat_info(save_pickle)
print("👍 compressed pickle size: {}".format(info.st_size))

# TODO Read totals.pickle to reconstruct all datasets and labels for next steps.
# For evaluation, use all from memory directly.

print("► reformat total.pickle.")
train_dataset, train_labels = training_helper.flat_dataset_labels(train_dataset, train_labels, 10)
valid_dataset, valid_labels = training_helper.flat_dataset_labels(valid_dataset, valid_labels, 10)
test_dataset, test_labels = training_helper.flat_dataset_labels(test_dataset, test_labels, 10)

print("👍 ")
print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)
