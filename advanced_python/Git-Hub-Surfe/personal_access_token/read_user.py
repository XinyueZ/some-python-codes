#
#Use oauth tokens to get user info on github.
#
from requests import get as http_get

target_url = "https://api.github.com/user"

def call_get_user(username, token):
    """
    Call github with token to get "/user".
    See. curl -u username:token https://api.github.com/user
    See. https://developer.github.com/v3/auth/#basic-authentication
    """
    res = http_get(target_url, auth = (username, token))
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



