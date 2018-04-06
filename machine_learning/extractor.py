#
# Util class Extractor provides extracting compressed objects.
#

DEBUG = True # Only for tests code below.

import sys
from os.path import splitext as split_text
from os.path import isdir as is_dir
from os.path import isfile as is_a_file
from os.path import join
from tarfile import open as open_compressed_object

class Extractor:
    def __init__(self, object_fullname_list, saved_data_root = "."):
        """
        Construct the Extractor with collection of objects with fullnames.
        The root directory where all objects would extracted into.
        """
        self.source_fullname_list = object_fullname_list
        self.data_root = saved_data_root


    def __extract_object__(self, source_object_fullname):
        """
        Extract object with source_object_fullname. This extractor extract object focely regardless of existing target.
        """
        with open_compressed_object(source_object_fullname) as tar_file:
            sys.stdout.flush() # Some stackoverflow answers suggest here for before extracting.
            print("► extracting: {}, it might take several minutes, please wait ❄.".format(source_object_fullname))
            tar_file.extractall(self.data_root)
            print("👍  finished")

    def extract(self):
        """
        Extract all objects which are pointed by source_fullname_list.
        """
        for source_object_fullname in self.source_fullname_list:
            if not is_a_file(source_object_fullname):
                print("☠  can't find {}.".format(source_object_fullname))
            else:
                self.__extract_object__(source_object_fullname)

if DEBUG:
    src_list = ["./notMNIST_large.tar.gz", "./notMNIST_small.tar.gz"]
    extractor = Extractor(src_list)
    extractor.extract()
