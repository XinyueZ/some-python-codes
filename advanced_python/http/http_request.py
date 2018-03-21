import requests

res = requests.get("http://www.online.sh.cn")
print(res.text)
print(type(res.text))
