import time
import webbrowser

DEBUG = True

def take_break(target_url, times = 2, wait_time_sec = 10):
    """
    Make cycle to open installed webbrowser.

    times: Int for open times of browser.
    wait_time_sec: Int duration for each opening.
    target_url: String which page to open.
    """

    for _ in range(times):
        if DEBUG: print("Wait for {} seconds.".format(wait_time_sec))
        time.sleep(wait_time_sec)
        if DEBUG: print("Open {}".format(target_url))
        webbrowser.open(target_url)
    print("Stop working, thanks for using.")


take_break(target_url = "http://www.youtube.com/watch?v=dQw4w9WgXcQ")


