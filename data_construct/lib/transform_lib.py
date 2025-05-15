# data_construct/transform_lib.py
# Python Standard Library
import random

def check_good(sssss, rule_dict):
    """
    Detect whether all characters in a single word are in the keys of the rule
    """
    alist = set(rule_dict.keys())
    for char in sssss:
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


def char_drift(c):
    # Shift each letter forward by one position in the alphabet, wrapping from 'z' back to 'a'
    if c.isupper():
        return chr((ord(c) - ord('A') + 1) % 26 + ord('A'))
    else:
        return chr((ord(c) - ord('a') + 1) % 26 + ord('a'))


def Transform_1(word):
    # Duplicate each string, for example, "happy" becomes "hhaappppyy"
    return "".join([c * 2 for c in word])


def Transform_2(word):
    # Shift each letter one position to the right in the alphabet. For example, "happy" becomes "ibqqz"
    return "".join([char_drift(c) for c in word])


def Transform_3(word):
    # Shift a string one position to the right, cyclically. For example, "happy" becomes "yhapp"
    return word[-1] + word[:-1] if word else word


def Transform_4(word):
    # Reverse a string, for example, "happy" becomes "yppah"
    return word[::-1]


def Transform_5(word):
    # Shift a string's characters two positions to the left; for example, "happy" becomes "ppyha"
    return word[2:] + word[:2] if len(word) > 2 else word


def Transform_6(word):
    # Rotate the letters in even positions. For example, "happy" becomes "hbpqy"
    return "".join([char_drift(c) if i % 2 == 0 else c for i, c in enumerate(word)])


def Transform_7(word):
    # Rotate the letters in odd positions. For example, "happy" becomes "iaqpz"
    return "".join([char_drift(c) if i % 2 == 1 else c for i, c in enumerate(word)])


def Transform_8(word, dic):
    # Convert letters into their corresponding emoji strings
    return "|".join([char2emoji(c.lower(), dic) for c in word])

def word_transform(word, dic):
    """
    Transform the word using a dictionary of transformations
    """
    # Define the transformation functions
    transformations = {
        "Transform_1": Transform_1,
        "Transform_2": Transform_2,
        "Transform_3": Transform_3,
        "Transform_4": Transform_4,
        "Transform_5": Transform_5,
        "Transform_6": Transform_6,
        "Transform_7": Transform_7,
        "Transform_8": lambda x: Transform_8(x, dic),
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
    transformed_word = word_transform(word, dic)
    print(transformed_word)
