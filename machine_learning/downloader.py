#
# Util class Downloader provides method to download object from net.
#

import sys
from os import stat
from os.path import exists as path_exists
from os.path import join as path_join
from urllib.request import urlopen as open_url

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

    def __progress__(self, count, block_size, total_size):
        """
        Report downloading progress.
        """
        percent = int(count * block_size * 100 / total_size)
        if self.last_percent != percent:
            print("⏬ {}%".format(percent), sep=' ',  end = "\r", flush = True)
        
        self.last_percent = percent

    def __get_object_size__(self, source_object_fullname):
        """
        Get object byte-size.
        """
        with open_url(source_object_fullname) as file:
            return int(file.getheader("Content-Length"))

    def download(self, object_name, force = False):
        """
        Verify whether the given object_name is available to download or not.
        Set force_download with True if the download execute although it exists.
        Return None when the object could not be downloaded. After downloading the 
        byte-size of downloaded object will be checked and verified.
        """
        dest_object_fullname = path_join(self.data_root, object_name)
        if force or not path_exists(dest_object_fullname):
            print("► download: {}.".format(object_name))
            source_object_fullname = self.url + object_name
            print("☁ source: {}.".format(source_object_fullname))
            urlretrieve(source_object_fullname, dest_object_fullname, reporthook = self.__progress__)
            print("\n👍 finished.")

            expected_bytes = self.__get_object_size__(source_object_fullname)
            object_stat_info = stat(dest_object_fullname)
            print("✄ verifying: {}.".format(object_name))
            if object_stat_info.st_size == expected_bytes:
                print("✅ verified.")          
                return dest_object_fullname
            else:
                print("☠  couldn't download {} and failed to verify {}.".format(object_name, dest_object_fullname))
                return None
        else:
            print("👍 {} already existed.".format(dest_object_fullname))   
            return dest_object_fullname




 
"""
Download objects
"""
downloader = Downloader('https://commondatastorage.googleapis.com/books1000/') 
train_filename = downloader.download('notMNIST_large.tar.gz')
test_filename = downloader.download('notMNIST_small.tar.gz')
