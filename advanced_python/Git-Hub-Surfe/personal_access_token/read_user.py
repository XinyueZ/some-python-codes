#
#Use oauth tokens to get user info on github.
#
from requests import get as http_get
from os import listdir as list_directory

DEBUG = True
target_url = "api.github.com/user"

def make_up_url(url, username, token):
    final_url = "https://{}:{}@{}".format(username.strip(), token.strip(), url.strip())
    
    if DEBUG:
        print("final_url = {}".format(final_url))
    
    return final_url

def call_get_user(username, token):
    """
    Call github with token to get "/user".
    See. curl -u username:token https://api.github.com/user
    See. https://developer.github.com/v3/auth/#basic-authentication
    """
    res = http_get(make_up_url(target_url, username, token))
    print(res.text)

def get_info(path):
    """
    Read token from file in path.
    Whole file will be read.
    """
    with open(path) as file:
        return file.read()


USER_NAME_FILE = "username"
TOKEN_FILE = "token"
INFO_FILE_LIST = list_directory(".")

if USER_NAME_FILE in INFO_FILE_LIST and TOKEN_FILE in INFO_FILE_LIST:
    username = get_info(USER_NAME_FILE)
    token = get_info(TOKEN_FILE)
    call_get_user(
            username,
            token
    )
else:
    print("Check out whether username is specified in file: [./{}] and token in file: [./{}]".format(USER_NAME_FILE, TOKEN_FILE))



