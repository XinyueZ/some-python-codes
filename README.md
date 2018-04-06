Some codes in Python(3)
===
A bucket for some codes in Python which are written at my free time at home. 
Some of them are from online-lessons, some of them are from my own creations.

# Purpose
No purpose only for fun. Only find that the codes must be saved in somewhere to 
prevent from losing for some special occasions.

# Spotlight

- [http crawl](https://github.com/XinyueZ/some-python-codes/tree/master/advanced_python/http_crawl)
	- A small web-crawl tool to crawl first href-link on wikipedia content. 
		- From a wiki-page to a target page.
		- Can define max continue if target page can not be reached.
- [send SMS](https://github.com/XinyueZ/some-python-codes/tree/master/advanced_python/send_message)
	- Use [Twilio](https://www.twilio.com/) to send SMS.

- [profanity content check](https://github.com/XinyueZ/some-python-codes/tree/master/advanced_python/profanity_check)
	- Use [Google's profanity API](http://www.wdylike.appspot.com/?q=some_content) to check plant text contents.
		- Edit [message.txt](https://github.com/XinyueZ/some-python-codes/blob/master/advanced_python/profanity_check/message.txt), just give some texts freely.
		- Find profanity or embarrassing content.
- [show movie trailer](https://github.com/XinyueZ/some-python-codes/tree/master/advanced_python/movie_trailer_website)
	- Show a website(opening system browser) on local computer with some film trailers.

# Machine Learning

Introduce a couple of python codes for machine-learning program. They're maybe used in some cases like data-source downloading,  data-source extract, serializing data-set etc. Some of these codes might use [Tensorflow](https://www.tensorflow.org/) as advanced sample.

I keep these code as simple as possible to catch and easy to understand for furture machine-learning programming(ML), because a lot ML codes, even Hello,World, will *start with downloading, construct data-source to data-set or batch* which is not cheap for newcomer, here give only a buffer step to check out what high-level library like [Tensorflow](https://www.tensorflow.org/) does for general job. Step-By-Step is sometimes powerful key to *Know-How* .  

- [Downloader, used for loading training data](https://github.com/XinyueZ/some-python-codes/tree/master/machine_learning/downloader.py).
	- run ```python3 downloader.py``` directly for sample call(Flag DEBUG = True).
- [Extractor, used for extract compressed file](https://github.com/XinyueZ/some-python-codes/tree/master/machine_learning/extractor.py).
	- run ```python3 extractor.py``` directly for sample call(Flag DEBUG = True).
- [PickleMaker, used for serializing object, i.e. images, to binary](https://github.com/XinyueZ/some-python-codes/tree/master/machine_learning/pickle_maker.py)
	- run ```python3 pickle_maker.py``` directly for sample call(Flag DEBUG = True).

# License

```
			MIT License

                Copyright (c) 2018 Chris Xinyue 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
