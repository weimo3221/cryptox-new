# Python Standard Library
import argparse
import os
import time
import random

# Commonly Used Open-Source Libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

# Libraries for projects
from llamawrapper import LlamaHelper
from prompt import logitlens_prompt

# Setting TOKENIZERS_PARALLELISM
os.environ['TOKENIZERS_PARALLELISM'] = "true"

def set_seed(seed: int = 719):
    """
    Reset seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True

# Reset seed
seed = 0
set_seed(seed=seed)

# Define and parse the command line arguments passed to logit_lens
parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default="Qwen2.5-7B-Instruct",help="The name of the model for the test needs to be the same as the one set up")
parser.add_argument("--model_path", "-p", type=str, default="Qwen2.5-7B-Instruct",help="The model path for testing needs to be the same as the one set up")
parser.add_argument("--dataset", "-d", type=str, default='base_morse',help="测试的数据集名称，需要和设置的一致")
parser.add_argument("--outputpath", "-o", type=str, default='base_morse',help="输出结果的路径，需要和设置的一致")
args = parser.parse_args()
model_size = args.model
encoded_type = args.dataset
custom_model = args.model_path
output_path = args.outputpath

# Load the custom model, tokenizer, and prepare the model's associated components and vocabulary mappings
print(f"\nStart loading model {model_size}...\n")
llama = LlamaHelper(dir=custom_model, load_in_4bit=True, device_map='auto')
tokenizer = llama.tokenizer
model = llama.model
unemb = nn.Sequential(llama.model.model.norm, llama.model.lm_head)
id2voc = {id:voc for voc, id in llama.tokenizer.get_vocab().items()}
voc2id = llama.tokenizer.get_vocab()
print(f"\nStart loading dataset...\n")

def token_prefixes(token_str: str):
    """
    Get all the prefixes of a given string, return a list of prefixes.
    """
    n = len(token_str)
    if n==1:
        tokens = [token_str[:i] for i in range(1, n+1)]
    else:
        tokens = [token_str[:i] for i in range(2, n+1)]
    return tokens 

def add_spaces(tokens):
    """
    Adds a space character to each token and returns
    """
    return ['Ġ' + t for t in tokens] + tokens
def capitalizations(tokens):
    """
    Processes and returns a unique collection of case-sensitive tokens
    """
    return list(set(tokens))

def unicode_prefix_tokid(zh_char = "积", tokenizer=tokenizer):
    """
    Finds the token id of a character based on its Unicode code
    """
    if not zh_char.encode().__str__()[2:-1].startswith('\\x'):
        return None
    start = zh_char.encode().__str__()[2:-1].split('\\x')[1]
    unicode_format = '<0x%s>'
    start_key = unicode_format%start.upper()
    if start_key in tokenizer.get_vocab():
        return tokenizer.get_vocab()[start_key]
    return None

def process_tokens(token_str: str, tokenizer):
    """
    Processes token strings, generating a list of token ids with prefixes, spaces, and case handling
    """
    with_prefixes = token_prefixes(token_str)
    with_spaces = add_spaces(with_prefixes)
    with_capitalizations = capitalizations(with_spaces)
    final_tokens = []
    for tok in with_capitalizations:
        if tok in tokenizer.get_vocab():
            final_tokens.append(tokenizer.get_vocab()[tok])
    tokid = unicode_prefix_tokid(token_str[0], tokenizer)
    if tokid is not None:
        final_tokens.append(tokid)
    
    if "Qwen" in model_size:
        with_prefixes = token_prefixes(token_str)
        with_spaces = add_spaces(with_prefixes)
        with_capitalizations = capitalizations(with_spaces)
        for tok in with_capitalizations:
            id = tokenizer(tok, return_tensors='pt')['input_ids'][0][0].item()
            final_tokens.append(id)
    
    final_tokens = list(set(final_tokens))
    return final_tokens

def pre_process(fin,target_rule):
    """
    Read and preprocess the dataset to generate a list of token ids based on the encoding method
    """
    df = pd.read_json(fin, lines=True)
    df = df.sample(min(800, len(df)), random_state=int(time.time()))
    dataset_gap = []
    
    for idx, item in tqdm(df.iterrows(), total=len(df), desc="Processing prompts"):   
        item_dict = item.to_dict()
        prompt = logitlens_prompt.format(**item_dict)
        decoded_token_id=[]
        out_token_id=[]
        #处理答案相关的tokens
        answer = item['answer']
        out_token_str = [item['answer']]
        if answer=="A":
            out_token_str+=item['choices_a'].split(" ")
        elif answer=="B":
            out_token_str+=item['choices_b'].split(" ")
            if "A" in out_token_str:
                out_token_str.remove("A")
        elif answer=="C":
            out_token_str+=item['choices_c'].split(" ")
            if "A" in out_token_str:
                out_token_str.remove("A")
        elif answer=="D":
            out_token_str+=item['choices_d'].split(" ")
            if "A" in out_token_str:
                out_token_str.remove("A")
        if target_rule[-1]=='0':
            for word in out_token_str:
                out_token_id+=process_tokens(word, tokenizer)            
            out_token_id=list(set(out_token_id))
            if len(out_token_id) == 0:
                continue
            dataset_gap.append({
                'prompt': prompt,
                'out_token_id': out_token_id,
            })
        else:
            #处理解码后单词相关的tokens
            for word in item['sample_word']:
                decoded_token_id += process_tokens(word, tokenizer)
            decoded_token_id=list(set(decoded_token_id))
            for word in out_token_str:
                out_token_id+=process_tokens(word, tokenizer)            
            out_token_id=list(set(out_token_id))
            if len(out_token_id) == 0:
                continue
            dataset_gap.append({
                'prompt': prompt,
                'out_token_id': out_token_id,
                'decoded_token_id': decoded_token_id,
            })

    return dataset_gap



def process_logit(target_rule):
    """
    Selecting dataset paths based on encoding rules and processing token probabilities
    """
    if target_rule == 'emoji_shuffle_0':
        dataset='data/emoji_shuffle/modified_emoji_shuffle_mmlu_dev_285_hop_1_percentage_0.jsonl'
    elif target_rule == 'emoji_shuffle_3':
        dataset='data/emoji_shuffle/modified_emoji_shuffle_mmlu_dev_285_hop_1_percentage_3.jsonl'
    elif target_rule == 'emoji_shuffle_5':
        dataset='data/emoji_shuffle/modified_emoji_shuffle_mmlu_dev_285_hop_1_percentage_5.jsonl'
    elif target_rule == 'emoji_morse_0':
        dataset = 'data/emoji_morse/modified_emoji_morse_mmlu_dev_285_hop_1_percentage_0.jsonl'
    elif target_rule == 'emoji_morse_3':
        dataset = 'data/emoji_morse/modified_emoji_morse_mmlu_dev_285_hop_1_percentage_3.jsonl'
    elif target_rule == 'emoji_morse_5':
        dataset = 'data/emoji_morse/modified_emoji_morse_mmlu_dev_285_hop_1_percentage_5.jsonl'
    elif target_rule == 'base_morse_0':
        dataset = 'data/morse/modified_morse_base_mmlu_dev_285_hop_1_percentage_0.jsonl'
    elif target_rule == 'base_morse_3':
        dataset = 'data/morse/modified_morse_base_mmlu_dev_285_hop_1_percentage_3.jsonl'
    elif target_rule == 'base_morse_5':
        dataset = 'data/morse/modified_morse_base_mmlu_dev_285_hop_1_percentage_5.jsonl'
    assert(dataset is not None)
    print(f"loading dataset:{dataset}")

    dataset_gap = pre_process(dataset,target_rule)
    print("Dataset Length: ", len(dataset_gap))
    out_token_probs = []
    decoded_token_probs = []
    all_possible_out_token_probs = []

    # Selecting dataset paths based on encoding rules and processing token probabilities
    for idx, d in enumerate(tqdm(dataset_gap, desc="Inference instance num")):
        prompt_tmp=[i for i in d['prompt']]
        prompt=["".join(prompt_tmp).strip()]
        latents = llama.latents_all_layers(prompt)
        latents = latents.to('cuda')
        logits = unemb(latents)
        last = logits[:, -1, :].float().softmax(dim=-1).detach().cpu()
        out_token_probs += [last[:, torch.tensor(d['out_token_id'])].sum(axis=-1)]
        if target_rule[-1]!='0':
            decoded_token_probs += [last[:, torch.tensor(d['decoded_token_id'])].sum(axis=-1)]
    if target_rule[-1]=='0':
        out_token_probs = torch.stack(out_token_probs)
        return out_token_probs
    else:
        out_token_probs = torch.stack(out_token_probs)
        decoded_token_probs = torch.stack(decoded_token_probs)
        return out_token_probs,decoded_token_probs
print(f"Using the model in {custom_model}")

# Call process_logit to perform LogitLens analysis on the dataset coded with 0/3/5 words
out_token_probs_0=process_logit(f'{encoded_type}_0')
out_token_probs_0 = out_token_probs_0.mean(dim=0)

out_token_probs_3,decoded_token_probs_3=process_logit(f'{encoded_type}_3')
out_token_probs_3 = out_token_probs_3.mean(dim=0)
decoded_token_probs_3 = decoded_token_probs_3.mean(dim=0)

out_token_probs_5,decoded_token_probs_5=process_logit(f'{encoded_type}_5')
out_token_probs_5= out_token_probs_5.mean(dim=0)
decoded_token_probs_5= decoded_token_probs_5.mean(dim=0)

# Draw an image and save it
fig, ax1 = plt.subplots(figsize=(10, 6))

line1, = ax1.plot(range(1, len(out_token_probs_0) + 1), out_token_probs_0, label=f'out_token_probs_0(answer)', color='tab:green')
line2, = ax1.plot(range(1, len(out_token_probs_3) + 1), out_token_probs_3, label=f'out_token_probs_3(answer)', color='tab:orange')
line3, = ax1.plot(range(1, len(out_token_probs_5) + 1), out_token_probs_5, label=f'out_token_probs_5(answer)', color='tab:blue')

ax1.set_xlabel('Layer')
ax1.set_ylabel('Probability (answer)')
ax1.set_title(f'{model_size} - {encoded_type} - Probability vs. Layer')

ax2 = ax1.twinx()
line4, = ax2.plot(range(1, len(decoded_token_probs_3) + 1), decoded_token_probs_3, label='out_token_probs_3 (decoded_words)', color='tab:purple', linestyle='--')
line5, = ax2.plot(range(1, len(decoded_token_probs_5) + 1), decoded_token_probs_5, label='out_token_probs_5 (decoded_words)', color='tab:red', linestyle='--')

if encoded_type=='emoji_shuffle':
    out_dir = f'{output_path}/emoji_shuffle'
    os.makedirs(f'{os.path.join(out_dir,model_size )}', exist_ok=True)
    plt.savefig(f'{os.path.join(out_dir,model_size )}/{model_size}_merged_plot.pdf', dpi=300, bbox_inches='tight')
elif encoded_type=='emoji_morse':
    out_dir = f'{output_path}/emoji_morse'
    os.makedirs(f'{os.path.join(out_dir,model_size )}', exist_ok=True)
    plt.savefig(f'{os.path.join(out_dir,model_size )}/{model_size}_merged_plot.pdf', dpi=300, bbox_inches='tight')
elif encoded_type=='base_morse':
    out_dir = f'{output_path}/base_morse'
    os.makedirs(f'{os.path.join(out_dir,model_size )}', exist_ok=True)
    plt.savefig(f'{os.path.join(out_dir,model_size )}/{model_size}_merged_plot.pdf', dpi=300, bbox_inches='tight')
plt.show()