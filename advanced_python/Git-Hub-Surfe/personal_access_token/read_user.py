#
#Use oauth tokens to get user info on github.
#
from requests import get as http_get

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


call_get_user(
            get_info("username"),
            get_info("token")
        )



