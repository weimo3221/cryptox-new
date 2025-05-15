# data_construct/encode_lib.py
# Python Standard Library
import random
import base64
import logging
import sys
import os
import hashlib

# Commonly Used Open-Source Libraries
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5, DES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes


# Configuring logging, setting the logging level and output format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


def check_good_cyber(sssss, rule_dict):
    """
    Detect whether all characters in a single word are in the keys of the rule
    """
    alist = set(rule_dict.keys())
    if len(sssss) > 20:
        return False
    for char in sssss:
        char = char.upper()
        if char not in alist:
            return False
    return True


def sample_idx_cyber(question, rule_dict, percent=0.2, encode_type="percentages"):
    """
    Get the idx of the word that needs to be encoded
    """
    # Setting the random seed
    random.seed(42)
    question_word_list = question.split(' ')
    question_word_sample = []
    question_word_sample_idx = []
    for idx, word in enumerate(question_word_list):
        is_good = check_good_cyber(word, rule_dict)
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


def morse_code(word, rule_dict):
    """
    Encoding of word
    """
    morse_code_list = []
    for char in word:
        char = char.upper()
        morse_code_list.append(rule_dict[char.upper()])
    morse_code = '|'.join(morse_code_list)
    return morse_code.strip()


def caesar_code(word, key):
    """
    Encoding of word using Caesar cipher
    """
    positive_shift= key
    morse_code_list = []
    for char in word:
        char = char.upper()
        morse_code_list.append(chr((ord(char) - ord("A") + positive_shift) % 25 + ord("A")))
    morse_code = ''.join(morse_code_list)
    return positive_shift, morse_code.strip()


def rsa_code(word, key):
    """
    Encoding of word using RSA cipher
    """
    key_length_bits = 512
    pem_data = open(key, 'rb').read()
    imported_key = RSA.import_key(pem_data)
    public_key = imported_key.publickey()

    word_bytes = word.encode('utf-8')
    if len(word_bytes) > (key_length_bits // 8 - 11):
        logging.error(f"The word: {word} is too long; it exceeds the {key_length_bits // 8 - 11}-byte limit.")
        sys.exit(1)

    cipher_rsa_encrypt = PKCS1_v1_5.new(public_key)
    encrypted_bytes = cipher_rsa_encrypt.encrypt(word_bytes)

    encrypted_b64_str = base64.b64encode(encrypted_bytes).decode('utf-8')
    
    return f"n: {imported_key.n}, e: {imported_key.e}, d: {imported_key.d}", encrypted_b64_str.strip()


def des_code(word, key):
    """
    Encoding of word using DES cipher; MODE: DES.ECB
    """
    key = get_random_bytes(8)
    word_bytes = word.encode('utf-8')
    cipher = DES.new(key, DES.MODE_ECB)

    padded_plaintext = pad(word_bytes, DES.block_size)
    ciphertext_bytes = cipher.encrypt(padded_plaintext)
    
    return key.hex(), ciphertext_bytes.hex().strip()

def md5_code(word, key=None):
    """
    Encoding of word using MD5 hash function
    """
    md5_hash = hashlib.md5()
    md5_hash.update(word.encode('utf-8'))
    return key, md5_hash.hexdigest().strip()

cyber_rule = {"rsa": rsa_code,
              "caesar": caesar_code, 
              "des": des_code, 
              "md5": md5_code}

if __name__ == "__main__":
    word = "hello"
    key = r"D:\GithubFiles\CryptoX-for-ICML\data_construct\lib\rsa_key_512.pem"
    print(rsa_code(word, key))