from os import listdir as list_directories
from os import rename as file_rename
from os import getcwd as pwd
from os import chdir as cd

def remove_filename_numbers(path):
    """
    Filenames might have numbers, this function is to remove them.
    """
    # Get list of filenames under path directory.
    dir_list = list_directories(path)
    # Ready to remove all numbers and replace them with empty.
    table = str.maketrans("0123456789", "          ", "0123456789")
    # Save current work directory because we returns here after changing filenames.
    work_dir = pwd()
    # Warning! In order to change filenames, we must go into the directory of files.
    cd(path)
    # Go to pictures and change filenames.
    for filename in dir_list:
        new_filename = filename.translate(table)
        print("{} -> {}".format(filename, new_filename))
        file_rename(filename, new_filename)
    cd(work_dir)


def print_filenames(path):
    """
    List filenames under path.
    """
    print(list_directories(path))



PICTURES_DIRECTORY = "prank"
remove_filename_numbers(PICTURES_DIRECTORY)
print_filenames(PICTURES_DIRECTORY)
