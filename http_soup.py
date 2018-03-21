import requests
from bs4 import BeautifulSoup 

res = requests.get("http://www.online.sh.cn")
soup = BeautifulSoup(res.text)

print(soup.prettify())
print("*" * 90)
print(soup.title)
print("#" * 90)
print(soup.body)
