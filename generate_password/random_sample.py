"""
This is another sample to generate password.
"""

import random
# Use an import statement at the top

word_file = "words.txt"
word_list = []

#fill up the word_list
with open(word_file,'r') as words:
    for line in words:
        # remove white space and make everything lowercase
        word = line.strip().lower()
        # don't include words that are too long or too short
        if 3 < len(word) < 8:
            word_list.append(word)


def generate_password(): return "".join(random.sample(word_list, 3))
                                                                    
# test your function
print(generate_password())
