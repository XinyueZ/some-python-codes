#
# Util class PickleMaker will convert objects under one folder
# to a collection of 2-D collection, a 3-D collection will be.
#

DEBUG = True # Test with codes below.

from imageio import imread as read_image
import numpy as np
from numpy import ndarray 
from numpy import std as standard_deviation
from numpy import mean as dataset_mean
from os.path import isdir as is_dir
from os import listdir as directory_list
from os.path import join as path_join
from six.moves import cPickle as pickle

class PickleMaker:
    def __init__(self, folder_with_objects_fullname_list, expected_objects_count, each_object_size_width = 28, each_object_size_height = 28,  pixel_depth = 255.0):
        """
        Construct the PickleMaker that can convert objects under folders of
        folder_with_objects_fullname_list to a collection of 2-D, a 3-D collection.
        A validated result collection should contain rows not less than expected_objects_count.
        Give each_object_size_width/height to avoid unsituable training objects.
        """
        self.folder_with_objects_fullname_list = folder_with_objects_fullname_list
        self.expected_objects_count = expected_objects_count
        self.each_object_size_width = each_object_size_width
        self.each_object_size_height = each_object_size_height
        self.pixel_depth = pixel_depth

    def __convert_object_to_dataset__(self, object_fullname):
        """
        Convert a object to 2-D, and return in 2-D array.
        Return None, False when some errors.
        """
        try:
            print("► convert {}.".format(object_fullname), sep=' ',  end = "\r", flush = True)
            dataset = (read_image(object_fullname).astype(float) - self.pixel_depth / 2) / self.pixel_depth
            # Only the object with size-width-height equals to what we expected should
            # be as dataset to be persistented lately.
            if dataset.shape == (self.each_object_size_width, self.each_object_size_height): 
                print("✄ {}".format(object_fullname), sep=' ',  end = "\r", flush = True)
                return dataset, True
            else:
                print("☠  {} won't be used as training data.".format(object_fullname), sep=' ',  end = "\r", flush = True)
                return None, False
        except (IOError, ValueError) as ex:
                print("☠  {} won't be used as training data.".format(object_fullname), sep=' ',  end = "\r", flush = True)
                return None, False

    def __convert_objects_to_dataset__(self, folder_fullname):
        """
        Loop objects under folder_fullname and transfer them to 2-D with
        __from_object_to_dataset__. 
        
        It'll save these 2-Ds to a 3-D collection which contains

        row: count of objects, 
        x: count of pixel of a object in  horizontal(width), 
        y: count of pixel of a object in vertical(height). 
        
        Return a 3-D as final result collection, otherwise None, False.
        """
        print("► convert objects from {}.".format(folder_fullname))
        object_fullname_list = directory_list(folder_fullname)
        return_dataset = ndarray(shape = (len(object_fullname_list), self.each_object_size_width, self.each_object_size_height), dtype = np.float32)
        count_converted = 0
        for obj in object_fullname_list:
            object_fullname = path_join(folder_fullname, obj)
            object_dataset, res = self.__convert_object_to_dataset__(object_fullname)
            if res:
                return_dataset[count_converted, :, :] = object_dataset
                count_converted += 1
    
        if count_converted >= self.expected_objects_count:    
            print("✄  filter useful data {}/{}.".format(count_converted, len(return_dataset)))
            return_dataset = return_dataset[:count_converted, :, :] # Optimizing, the return_dataset can't be used totally, the rest empty will be discared.
            print("✅  full-tensor: {}, mean: {}, std.deviation: {}".format(return_dataset.shape, dataset_mean(return_dataset), standard_deviation(return_dataset)))
            return return_dataset, True
        else:
            print("☠  varlidated data is too less, expected: {}, real: {}",  self.expected_objects_count, count_converted)
            return None, False

    def make(self):
        """
        Make pickle files to persist dataset(arrays) from objects under folder_with_objects_fullname_list.
        Return list of output fullname.
        """
        output_fullname_list = []
        for folder_fullname in self.folder_with_objects_fullname_list:
             if is_dir(folder_fullname):
                output_fullname = "{}.pickle".format(folder_fullname)
                output_fullname_list.append(output_fullname)
                dataset, res = self.__convert_objects_to_dataset__(folder_fullname)
                if res:
                    try:
                        with open(output_fullname, "wb") as write_file:
                            print("► convert from {} to {}".format(folder_fullname, output_fullname))
                            pickle.dump(dataset, write_file, pickle.HIGHEST_PROTOCOL)
                            print("👍  finished.")
                    except Exception as ex:
                        print("☠  error while saving data to {}\ncased by: \n\n{}".format(output_fullname, ex))

        return output_fullname_list

def test(src_root, expected_count):
    """
    src_root contains all prepared objects under their folders.
    expected_count for expect output after convert.
    """
    li = directory_list(src_root)
    input_objects = []

    for input_data_name in li:
        input_objects.append(path_join(src_root, input_data_name))
        
    maker = PickleMaker(input_objects, expected_count)
    maker.make()

if DEBUG: 
    test("./notMNIST_large", 45000)
    test("./notMNIST_small", 1800)

