# eval/eval.py
# Python Standard Library
import asyncio
import os
import logging

# Commonly Used Open-Source Libraries
import pandas as pd
from tqdm import tqdm

# Libraries for projects
from .utils import parse_init
from .eval_lib import predict, metric, save_process


# Configuring logging, setting the logging level and output format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def eval_file(file_path, output_dir, model_name, address, key, sem):
    """
    Evaluation of individual documents
    """
    df = pd.read_json(file_path, lines=True)
    df = df.sample(len(df), random_state=42)
    item_list = []
    for idx, item in tqdm(df.iterrows(), total=len(df), desc="Processing prompts......"):
        item = item.to_dict()
        item_list.append(item)
    
    item_list = await predict(item_list, sem, model_name, address, key)
    item_list = await metric(item_list, sem)

    file_name = os.path.basename(file_path)
    save_process(item_list, output_dir, file_name)
    logging.info(f"Complete the evaluation of the file: {file_name}")



async def eval_files(dir_path, output_dir, model_name, address, key, sem):
    """
    Evaluating multiple files in a single directory
    """
    files = os.listdir(dir_path)
    for file in files:
        logging.info(f"Evaluating file: {file}")
        file_path = os.path.join(dir_path, file)
        if not os.path.exists(os.path.join(output_dir, file)):
            await eval_file(file_path, output_dir, model_name, address, key, sem)
        else:
            logging.info(f"File {file} has been evaluated, skip it")


async def main():
    """
    The main block of code, which performs the evaluation of the data, including calling the model and evaluating the response
    """
    sem = asyncio.Semaphore(2)  # Put the semaphore here.
    args = parse_init()
    model_name = args.model
    address = args.address
    key = args.key
    if args.recursive:
        await eval_files(args.input, args.output, model_name, address, key, sem)
    else:
        await eval_file(args.input, args.output, model_name, address, key, sem)


if __name__ == "__main__":
    asyncio.run(main())