# Size of our test object, i.e image with TRAIN_OBJECT_WIDTH X TRAIN_OBJECT_HEIGHT
TRAIN_OBJECT_WIDTH, TRAIN_OBJECT_HEIGHT = 28, 28

# How many classes to classify.
CLASSES_TO_TRAIN = 10

# How many data should be downloaded expectly.
EXPECTED_COUNT_TRAIN_DATA, EXPECTED_COUNT_TEST_DATA = 45000, 1800

# How many training objects should be used and their valid collections.
TRAIN_SIZE, TRAIN_VAILD_SIZE = 200000, 10000

# How many testing objects should be used and their valid collections.
TEST_TRAIN_SIZE = 10000


positions_of_collection_to_delete = [
    4, 6, 66, 34, 67, 22, 2, 11, 45, 62, 3, 41, 42, 47, 78, 44, 53, 23, 99, 91, 93, 36, 77, 79, 70, 30, 11, 14]
