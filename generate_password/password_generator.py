from random import randint as rdInt

word_file = "words.txt"
word_list = []

def find_word(lines, count_lines):
    line_number = rdInt(0, count_lines)
    word = lines[line_number].strip().lower()

    if 3 < len(word) < 8: return word
    else: return find_word(lines, count_lines)

def generate_password():
    """
    selects three random words from the list of words word_list 
    and concatenates them into a single string. Your function 
    should not accept any arguments and should reference the 
    global variable word_list to build the password.
    """
    random_times = 3
    with open(word_file) as file:
        lines = file.readlines()
        count_lines = len(lines)

        for _ in range(random_times):
            word_list.append(find_word(lines, count_lines - 1))
    
    return "".join(word_list)

print(generate_password())
