#
# Util class with some download functions which could be used in ML
# program to fetch remote objects for training.
#
#

import sys
from os import stat
from os.path import join as path_join
from os.path import exists as path_exists
from six.moves.urllib.request import urlretrieve

class Downloader:
    def __init__(self, src_url, saved_data_root = "."):
        """
        Construct the Downloader with src_url which points to the download source,
        saved_data_root which points where to save all downloaded objects.
        """
        self.url = src_url
        self.last_percent = None
        self.data_root = saved_data_root

    def progress(self, count, block_size, total_size):
        """
        Report downloading progress.
        """
        percent = int(count * block_size * 100 / total_size)
        if self.last_percent != percent:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()
        
        self.last_percent = percent

    def download(self, object_name, expected_bytes = None, force_download = False):
        """
        Verify whether the given object_name is available to download or not.
        Set force_download with True if the download execute although it exists.
        Return None when the object could not be downloaded. After downloading the 
        byte-size of downloaded object will be checked if expected_bytes is given.
        """
        dest_object_fullname = path_join(self.data_root, object_name)
        if force_download or not path_exists(dest_object_fullname):
            print("Trying to download: {}.".format(object_name))
            source_object_fullname = self.url + object_name
            print("Source: {}.".format(source_object_fullname))
            urlretrieve(source_object_fullname, dest_object_fullname, reporthook = self.progress)
            print("\n(¶) Finished download.")

            if expected_bytes == None:
                print("(!) Found {} but is not verified.".format(dest_object_fullname))
                return dest_object_fullname
            else:
                object_stat_info = stat(dest_object_fullname)
                print("Verifying object: {}.".format(object_name))
                if object_stat_info.st_size == expected_bytes:
                    print("(✓) Verified.")
                    return dest_object_fullname
                else:
                    print("(✘) Couldn't download {} and failed to verify {}.".format(object_name, dest_object_fullname))
                    return None
        else:
            return dest_object_fullname




"""
Download objects
"""
downloader = Downloader('https://commondatastorage.googleapis.com/books1000/') 
train_filename = downloader.download('notMNIST_large.tar.gz', 247336696)
test_filename = downloader.download('notMNIST_small.tar.gz', 8458043)
