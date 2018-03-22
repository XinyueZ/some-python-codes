import time
import requests
from bs4 import BeautifulSoup

def continue_crawl(recent_url_list, target_url, max_url_list_len = 25):
    """
    Returns True or False following these rules:

    If the most recent article in the recent_url_list is the target article the search should stop and the function should return False
    If the list is more than 25 urls long, the function should return False
    If the list has a cycle in it, the function should return False
    otherwise the search should continue and the function should return True.
    """
    most_recent = recent_url_list[-1]
    url_list_len = len(recent_url_list)
    url_list_in_set_len = len(set(recent_url_list))

    # The "set" of recent urls should be equal to "list" of recent, if not, 
    # that means there're some duplicated urls(cycle). 
    if url_list_len != url_list_in_set_len: return False
    if most_recent == target_url: return False
    if url_list_len > max_url_list_len: return False
    if target_url in recent_url_list: return False

    return True


print(continue_crawl(
    ['https://en.wikipedia.org/wiki/Floating_point', 'https://en.wikipedia.org/wiki/Computing', 'https://en.wikipedia.org/wiki/Floating_point'],
    'https://en.wikipedia.org/wiki/Philosophy') == False)
print(continue_crawl(["http://g.cn"], "http://sina.com.cn") == True)
print(continue_crawl(["http://g.cn"], "http://g.cn") == False)
print(continue_crawl(["http://g.cn", "http://www.online.sh.cn"], "http://g.cn") == False)
print(continue_crawl(["http://g.cn", "http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn"], "http://www.online.sh.ch") == False)

def find_first_link(url):
    """
    Get the HTML from "url", use the requests library
    feed the HTML into Beautiful Soup
    find the first link in the article
    
    return the first link as a string, or return None if there is no link.
    """

    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    article_link = soup.find(id = "mw-content-text").find(class_ = "mw-parser-output").p.a.get("href")
    if article_link: return article_link

print(find_first_link("https://en.wikipedia.org/wiki/A.J.W._McNeilly"))

def web_crawl(wait_sec = 2):
    while continue_crawl(article_chain, target_url):
        first_link = find_first_link(article_chain[-1])
        article_chain.append(first_link)
        time.sleep(wait_sec)


