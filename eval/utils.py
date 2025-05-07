# eval/utils.py
# Python Standard Library
import argparse
import logging
import os
import sys


# Configuring logging, setting the logging level and output format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def check_directory(path, mode):
    """
    Check the existence of input and output directories
    """
    if not os.path.isdir(path) and mode == "o":
        os.makedirs(path)
        return True
    elif not os.path.isdir(path) and mode == "i":
        return False
    else:
        return True
    

def check_file(path):
    """
    Check if the input file exists
    """
    if os.path.isdir(path):
        logging.error(f"Please add the -r argument to the command")
        sys.exit(1)
    elif os.path.exists(path):
        logging.info(f"Input file: {path}")
    else:
        logging.error(f"Please enter the correct file address")
        sys.exit(1)


def parse_init():
    """
    Defines and parses command line arguments to the eval code, configures logging, and checks for the existence of the input data file directory and the output directory
    """
    parser = argparse.ArgumentParser(description="Data creation utility")

    # Adding command line parameters
    parser.add_argument('-i', '--input', type=str, default="data_construct/crypto_data", help='file or directory to be evaluated')
    parser.add_argument('-r', '--recursive', action='store_true', help='If you need to evaluate all jsonl files in a directory, you need to add the -r directive and specify the input folder')
    parser.add_argument('-o', '--output', type=str, default="eval/result_data", help='Output path of the evaluation')
    parser.add_argument("-m", "--model", type=str, required=True, default="Qwen2.5-1.5B-Instruct", help="The name of the model for the test needs to be the same as the one set up")
    parser.add_argument("-a", "--address", type=str, required=True, default="http://localhost:9002/v1", help="The model address for the test needs to be the same as the one set up")
    parser.add_argument("-k", "--key", type=str, required=True, default="EMPTY", help="The key of the API")

    # Parsing command line parameters
    args = parser.parse_args()

    if args.recursive:
        if check_directory(args.input, "i"):
            logging.info(f"Input directory: {os.path.abspath(args.input)}")
        else:
            logging.error(f"Input directory is not exists: {os.path.abspath(args.input)}")
            sys.exit(1)
    else:
        check_file(args.input)

    if check_directory(args.output, "o"):
        logging.info(f"Output directory: {os.path.abspath(args.output)}")

    return args