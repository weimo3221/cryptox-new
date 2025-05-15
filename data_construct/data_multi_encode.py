# data_construct/data_multi_encode.py
# Python Standard Library
import os
import logging
import math

# Commonly Used Open-Source Libraries
import pandas as pd
from tqdm import tqdm
from Crypto.Random import get_random_bytes

# Libraries for projects
from .utils import parse_init_data_encode_multi, get_rule
from .lib.transform_lib import word_transform
from .lib.noise_lib import sample_idx, word_add_only_noise
from .lib.encode_lib import des_code

from .prompt_create.prompt_multi import multi_rule, multi_template, multi_template_math, multi_template_mbpp


# Configuring logging, setting the logging level and output format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def word_transform_all(words, rule, c_n, key):
    """
    Transforming a word according to the given rule
    """
    if c_n >= 1:
        words = word_add_only_noise(words, 0)
        new_key = key
    if c_n >= 2:
        new_key, words = des_code(words, key)
    if c_n >= 3:
        words = word_transform(words, rule)
    return new_key, words

def one_file_transform(file, df, math_rule, rule, percentage, output_dir_path, encode_type, c_n):
    """
    Transforming a dataset by percentage
    """
    fout = os.path.join(output_dir_path, f"{file.replace('.jsonl', '')}_percentage_{percentage}.jsonl")
    if math.isclose(percentage, 0, rel_tol=1e-9):
        logging.info(f"0 percentage is not supported, skipping {file.replace('.jsonl', '')}_percentage_{percentage}.jsonl......")
        return
    data = []
    for idx, item in df.iterrows():
        item = item.to_dict()
        question = item["question"]
        if item.get("dataset_type", "no_type") == "math_500":
            rule_chosen = math_rule
        else:
            rule_chosen = rule
        crypto_idx = sample_idx(question, rule_chosen["rule"], percentage, encode_type)
        crypto_word = []
        sample_word = []
        
        if c_n >= 2:
            key = get_random_bytes(8)
        else:
            key = None

        words = question.split(' ')
        for j in range(len(words)):
            if j in crypto_idx:
                sample_word.append(words[j])
                # print(f"words: {words[j]}")
                new_key, words[j] = word_transform_all(words[j], rule_chosen["rule"], c_n, key)
                crypto_word.append(words[j])
        
        crypto_question =' '.join(words)
        crypto_word_count = len(crypto_word)

        item["question"] = crypto_question
        item["ori_question"] = question
        item["rule"] = rule_chosen["prompt_rule"]
        item["crypto_word_count"] = crypto_word_count
        item['crypto_word'] = crypto_word
        item['sample_word'] = sample_word
        item["crypto_id"] = f'crypto_{str(idx)}'
        item["key"] = new_key
        item["tag"] = item["tag"] = os.path.basename(fout)
        if c_n >= 1:
            item["multi_rule"] = multi_rule["noise"]
        if c_n >= 2:
            item["multi_rule"] += multi_rule["des"]
        if c_n >= 3:
            item["multi_rule"] += multi_rule["transform"]

        if item.get("dataset_type", "no_type") == "math_500":
            item["prompt"] = [{"content": multi_template_math.format(**item).strip(), "role": "user"}]
        elif item.get("dataset_type", "no_type") == "mbpp_427":
            item["prompt"] = [{"content": multi_template_mbpp.format(**item).strip(), "role": "user"}]
        else:
            item["prompt"] = [{"content": multi_template.format(**item).strip(), "role": "user"}]

        data.append(item)
    df = pd.DataFrame(data)
    df.to_json(fout, orient='records', lines=True, force_ascii=False)
    

def files_transform(math_rule, rule, input_dir_path, percentages, output_dir_path, domain, encode_type, c_n):
    """
    Transforming multiple datasets, call one_file_transform to complete transforming
    """
    logging.info(f"Transforming data in directory: {input_dir_path}......")
    files = os.listdir(input_dir_path)
    for file in files:
        if not any(element in file for element in domain):
            continue
        file_path = os.path.join(input_dir_path, file)
        df = pd.read_json(file_path, lines=True)
        logging.info(f"Transforming data in file: {file}......")
        for percentage in percentages:
            logging.info(f"Transforming data in file: {file} with {encode_type[:-1]}: {percentage}")
            one_file_transform(file, df, math_rule, rule, percentage, output_dir_path, encode_type, c_n)


def main():
    """
    The main code block, which performs the transforming operations of the data set
    """
    args = parse_init_data_encode_multi()

    if args.rule in ["morse_base", "emoji_morse", "emoji_shuffle", "emoji_shuffle_new"]:
        rule = get_rule(args.rule)
        math_rule = get_rule(args.rule+"_math")
    else:
        rule = get_rule(args.rule)
        math_rule = get_rule(args.rule)
    input_dir_path = os.path.abspath(args.input)
    output_dir_path = os.path.abspath(args.output)
    if args.encode_type == "percentages":
        perentages = args.percentages
    else:
        perentages = args.nums
    domain = args.domain
    encode_type = args.encode_type
    c_n = args.combination

    files_transform(math_rule, rule, input_dir_path, perentages, output_dir_path, domain, encode_type, c_n)
    logging.info("Transforming is complete")


if __name__ == "__main__":
    main()