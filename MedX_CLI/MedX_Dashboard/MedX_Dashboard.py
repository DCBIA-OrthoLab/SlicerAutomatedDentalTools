#!/usr/bin/env python-real
import argparse
import torch
import sys, os, time

import logging

# ===== Logging Configuration =====
logger = logging.getLogger("MedX_dashboard_CLI")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

fpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(fpath)

from MedX_CLI_utils import show_dashboard


def main(args):
    os.makedirs(args.output_folder, exist_ok=True)
    
    show_dashboard(args.summary_folder, args.output_folder)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("summary_folder", type=str)
    parser.add_argument("output_folder", type=str)
    parser.add_argument("log_path", type=str)

    args = parser.parse_args()

    main(args)
