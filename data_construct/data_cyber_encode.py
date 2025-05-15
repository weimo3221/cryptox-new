# data_construct/data_cyber_encode.py
# Python Standard Library
import os
import logging
import math
import random
import copy

# Commonly Used Open-Source Libraries
import pandas as pd
from tqdm import tqdm
from Crypto.Random import get_random_bytes

# Libraries for projects
from .utils import parse_init_data_cyber_encode, get_rule
from .lib.encode_lib import sample_idx_cyber, cyber_rule
from .prompt_create.prompt_cyber import cyber_template, cyber_template_math, cyber_template_mbpp

cyber_prompt = {"rsa": "- The chosen word is encoded into bytes using UTF-8.\n- Next, these bytes are RSA encrypted using PKCS1_v1_5 padding, resulting in encrypted bytes.\n- The encrypted bytes are first Base64 encoded and then UTF-8 decoded into a string.\n- The RSA private key is as follows:\n{key}",
              "des": "- The chosen word is encoded into bytes using UTF-8.\n- Next, these bytes are padded to a length of 8 bytes using PKCS7 padding before being encrypted with DES, resulting in encrypted bytes.\n- The encrypted bytes will be directly converted into a hexadecimal string representation.\n- The DES key is: {key}",
              "md5": "- The chosen word is encoded into bytes using UTF-8.\n- Next, calculate the MD5 hash of these bytes, resulting in a 128-bit hash value.\n- The hash value is then converted into a hexadecimal string representation.",
              "caesar": "- Each character in the chosen word will be encrypted using a Caesar cipher with a shift of {key}"}

# Configuring logging, setting the logging level and output format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def one_file_encode(file, df, judge_rule, rule, percentage, output_dir_path, encode_type):
    """
    Encoding a dataset by percentage
    """
    fout = os.path.join(output_dir_path, f"{file.replace('.jsonl', '')}_percentage_{percentage}.jsonl")
    ori_rule = copy.deepcopy(judge_rule["rule"])
    
    data = []
    for idx, item in df.iterrows():
        item = item.to_dict()
        question = item["question"]
        if not item.get("dataset_type", "no_type") == "math_500":
            judge_rule = {k:v for k, v in ori_rule.items() if k.isalnum()}
        else:
            judge_rule = ori_rule

        crypto_idx = sample_idx_cyber(question, judge_rule, percentage, encode_type)
        crypto_word = []
        sample_word = []
        dir = os.path.dirname(os.path.abspath(__file__))

        # get the key for the encryption algorithm
        if rule == "rsa":
            os.system(f"openssl genpkey -algorithm RSA -out {os.path.join(dir, 'lib', f'rsa_key_512.pem')} -pkeyopt rsa_keygen_bits:512 > nul 2>&1")
            key = os.path.join(dir, "lib", f'rsa_key_512.pem')
        elif rule == "des":
            key = get_random_bytes(8)
        elif rule == "caesar":
            key = random.randint(1, 25)
        else:
            key = None

        cyber_mode = cyber_rule[rule]
        words = question.split(' ')
        if not math.isclose(percentage, 0, rel_tol=1e-9):
            for j in range(len(words)):
                if j in crypto_idx:
                    sample_word.append(words[j])
                    new_key, words[j] = cyber_mode(words[j], key)
                    crypto_word.append(words[j])
        
        else:
            new_key, _  =  cyber_mode("hello", key)
        
        crypto_question =' '.join(words)
        crypto_word_count = len(crypto_word)

        item["question"] = crypto_question
        item["ori_question"] = question
        item["rule"] = rule
        item["crypto_word_count"] = crypto_word_count
        item['crypto_word'] = crypto_word
        item['sample_word'] = sample_word
        item["crypto_id"] = f'crypto_{str(idx)}'

        item["cyber_rule"] = cyber_prompt[rule].format(key=new_key)
        item["tag"] = item["tag"] = os.path.basename(fout)

        if item.get("dataset_type", "no_type") == "math_500":
            item["prompt"] = [{"content": cyber_template_math.format(**item).strip(), "role": "user"}]
        elif item.get("dataset_type", "no_type") == "mbpp_427":
            item["prompt"] = [{"content": cyber_template_mbpp.format(**item).strip(), "role": "user"}]
        else:
            item["prompt"] = [{"content": cyber_template.format(**item).strip(), "role": "user"}]

        data.append(item)
    df = pd.DataFrame(data)
    df.to_json(fout, orient='records', lines=True, force_ascii=False)
    

def files_encode(judge_rule, rule, input_dir_path, percentages, output_dir_path, domain, encode_type):
    """
    Encoding multiple datasets, call one_file_encode to complete encoding
    """
    logging.info(f"Encoding data in directory: {input_dir_path}......")
    files = os.listdir(input_dir_path)
    for file in files:
        if not any(element in file for element in domain):
            continue
        file_path = os.path.join(input_dir_path, file)
        df = pd.read_json(file_path, lines=True)
        logging.info(f"Encoding data in file: {file}......")
        for percentage in percentages:
            logging.info(f"Encoding data in file: {file} with {encode_type[:-1]}: {percentage}")
            one_file_encode(file, df, judge_rule, rule, percentage, output_dir_path, encode_type)


def main():
    """
    The main code block, which performs the coding operations of the data set
    """
    args = parse_init_data_cyber_encode()

    
    input_dir_path = os.path.abspath(args.input)
    output_dir_path = os.path.abspath(args.output)
    if args.encode_type == "percentages":
        perentages = args.percentages
    else:
        perentages = args.nums
    domain = args.domain
    encode_type = args.encode_type
    rule = args.rule
    judge_rule = get_rule("emoji_shuffle_math")

    files_encode(judge_rule, rule, input_dir_path, perentages, output_dir_path, domain, encode_type)
    logging.info("Encoding is complete")


if __name__ == "__main__":
    main()