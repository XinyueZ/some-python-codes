import time
import requests
from bs4 import BeautifulSoup
from urllib.parse  import urljoin

MAX_LIST = 40
CRAWL_WAIT = 2 #seconds
DEBUG = True
DEFAULT_START = "https://en.wikipedia.org/wiki/Special:Random"
DEFAULT_END = "https://en.wikipedia.org/wiki/Philosophy"

def continue_crawl(recent_url_list, target_url, max_url_list_len = MAX_LIST):
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


def find_first_link(url, default_link):
    """
    Get the HTML from "url", use the requests library
    feed the HTML into Beautiful Soup
    find the first link in the article
    
    return the first link as a string, or return None if there is no link.
    """
    first_link = None
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    content_div = soup.find(id = "mw-content-text").find(class_ = "mw-parser-output")
    all_p_in_content = content_div.find_all("p", recursive = False)
    for child_p in all_p_in_content:
        # a_href = child_p.a # Wrong!, it will fetch nearst <a href>, also a child of other tags.
        a_href = child_p.find("a", recursive = False) # Best case.
        # a_href = (child_p.find_all("a", recursive = False))[0] # Not good, find.all returns collection.
        if a_href:
            first_link = a_href.get("href")
            first_link = urljoin(default_link, first_link)
            break
    return first_link

def web_crawl(start_url, target_url, wait_sec = CRAWL_WAIT):
    article_chain = [start_url]
    while continue_crawl(article_chain, target_url):
        first_link = find_first_link(article_chain[-1], start_url)
        if first_link:
            article_chain.append(first_link)
            if DEBUG: print("+ {}".format(first_link))
            time.sleep(wait_sec)
        else: return "We've reached a page where I cannot find a first-link, abort!"

    return article_chain

print("====BEGIN TESTS====")
print(find_first_link("https://en.wikipedia.org/wiki/Philosophy", "https://en.wikipedia.org/wiki/A.J.W._McNeilly"))
print(find_first_link("https://en.wikipedia.org/wiki/Philosophy", "https://en.wikipedia.org/wiki/Masatoshi_Nakayama"))
print(continue_crawl(
    ['https://en.wikipedia.org/wiki/Floating_point', 'https://en.wikipedia.org/wiki/Computing', 'https://en.wikipedia.org/wiki/Floating_point'],
    'https://en.wikipedia.org/wiki/Philosophy') == False)
print(continue_crawl(["http://g.cn"], "http://sina.com.cn") == True)
print(continue_crawl(["http://g.cn"], "http://g.cn") == False)
print(continue_crawl(["http://g.cn", "http://www.online.sh.cn"], "http://g.cn") == False)
print(continue_crawl(["http://g.cn", "http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn","http://g.cn"], "http://www.online.sh.ch") == False)

print("crawl page: https://en.wikipedia.org/wiki/Masatoshi_Nakayama -----> https://en.wikipedia.org/wiki/Philosophy")
print(web_crawl("https://en.wikipedia.org/wiki/Masatoshi_Nakayama", "https://en.wikipedia.org/wiki/Philosophy"))
print("crawl page: {} -----> {}".format(DEFAULT_START, DEFAULT_END))
print(web_crawl(DEFAULT_START, DEFAULT_END))
print("====END TESTS====")

