# data_construct/transform_lib.py
# Python Standard Library
import random
import string
import logging

# Configuring logging, setting the logging level and output format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_good(word, rule_dict):
    """
    Detect whether all characters in a single word are in the keys of the rule
    """
    alist = set(rule_dict.keys())
    for char in word:
        char = char.upper()
        if char not in alist:
            return False
    return True


def sample_idx(question, rule_dict, percent=0.2, encode_type="percentages"):
    """
    Get the idx of the word that needs to be encoded
    """
    # Setting the random seed
    random.seed(42)
    question_word_list = question.split(' ')
    question_word_sample = []
    question_word_sample_idx = []
    for idx, word in enumerate(question_word_list):
        is_good = check_good(word, rule_dict)
        if not is_good:
            continue
        elif len(word) <= 1:
            continue
        elif word in question_word_sample:
            continue
        else:
            question_word_sample_idx.append(idx)
            question_word_sample.append(word)
    if encode_type == "percentages":
        question_word_sample_idx = random.sample(question_word_sample_idx, int(percent * len(question_word_sample_idx)))
    elif encode_type == "nums":
        question_word_sample_idx = random.sample(question_word_sample_idx, min(int(percent), len(question_word_sample_idx)))
    return question_word_sample_idx


def char2emoji(c, dic):
    # Convert letters into their corresponding emoji strings
    new_mapping = {k.lower(): v for k, v in dic.items()}
    return new_mapping.get(c)


def Transform_1(word, type):
    # add noises
    letters = string.ascii_lowercase
    if type == 0:
        # 0 means add random letters to the odd bits of the word
        result = ""
        for i, c in enumerate(word):
            result += c
            if i % 2 == 0:
                result += random.choice(letters)
        return result
    elif type == 1:
        # 1 means randomly insert a letter into the word
        position = random.randint(0, len(word))
        return word[:position] + random.choice(letters) + word[position:]
    else:
        raise logging.error("Invalid transformation type. Use 0 for duplication or 1 for drift.")
    

def Transform_2(word, dic):
    # Convert letters into their corresponding emoji strings
    return "|".join([char2emoji(c.lower(), dic) for c in word])


def word_add_only_noise(word, type):
    return Transform_1(word, type)


def word_add_noise(word, dic, type):
    """
    Transform the word using a dictionary of transformations
    """
    # Define the transformation functions
    transformations = {
        "Transform_1": lambda x: Transform_1(x, type),
        "Transform_2": lambda x: Transform_2(x, dic),
    }

    for rule in transformations:
        # Apply each transformation to the word
        word = transformations[rule](word)
        # print(f"After {rule}: {word}")

    return word

if __name__ == "__main__":
    # Example usage
    word = "happy"
    dic = {"A": "😀🍎🚗", "B": "🐶🌟📚", "C": "🌈🍀🚀", "D": "🐱🍉🏀", "E": "🍔🎉🎈", "F": "🌸🍩🏰",
                "G": "🦋🍇⚽", "H": "🍕🎂🏝️", "I": "🍦🎁🎧", "J": "🐸🍒🏆", "K": "🦄🍓🎮", "L": "🐰🍍📷",
                "M": "🌹🍌🎨", "N": "🐼🍎🎤", "O": "🍉🎵📚", "P": "🌼🍇🎬", "Q": "🐢🍓🎯", "R": "🍒🎸📱",
                "S": "🌻🍍🎲", "T": "🐯🍌🎮", "U": "🍓🎹📖", "V": "🌺🍉🎥", "W": "🐳🍎🎭", "X": "🍍🎤📡",
                "Y": "🐥🍇🎨", "Z": "🌵🍒🎮", "0": "🦁🍄🎈", "1": "🐘🌸🎨", "2": "🦊🍉🌟",
                "3": "🦋🍇🎵", "4": "🦄🍓🎲", "5": "🐼🍍🎯", "6": "🦉🍌🎤", "7": "🦇🍒🎧",
                "8": "🦄🍂🎥", "9": "🦍🍁🎮"}
    transformed_word = word_add_noise(word, dic, 0)
    print(transformed_word)
