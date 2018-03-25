"""
Read file "message.txt" and check content of file 
about profanity or embarrassing words.

With help of  http://www.wdylike.appspot.com/?q=some_content
we can check content.

Find "True" in result means everything OK.
Find "False" in result means profanity or embarrassing.
"""
from urllib.request import urlopen as url_open
from urllib.parse import quote as url_quote

DEBUG = False 
api = "http://www.wdylike.appspot.com/?q={}"

def read_file(path):
    """
    Read file content with path.
    """
    with open(path) as file:
        return file.read()

def is_profanity(content):
    """
    Check whether content contains profanity information or not.
    Returns True for embarrassing status.
    """
    url = api.format(url_quote(content))

    if DEBUG:
        print("API: {}".format(url))

    with url_open(url) as cnn:
        result = str(cnn.read())

        if DEBUG:
            print("result text: {}\n\n".format(result))

        if "true" in result:
            return True
        if "false" in result:
            return False
        return None

def check_out():
    """
    Run check logical.
    """
    file_name = "message.txt"
    content = read_file(file_name)
    
    if DEBUG:
        print("File: {}-->\n\n{}".format(file_name, content))

    profanity = is_profanity(content)
    
    if profanity == None:
        print("Something wrong while calling API")
    else:
        if profanity:
            print("Bad, {} contains embarrassing words, please check it.".format(file_name))
        else:
            print("OK, {} is fine!".format(file_name))

check_out()

