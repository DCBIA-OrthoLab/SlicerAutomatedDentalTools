#!/usr/bin/env python-real
import argparse
import torch
import sys, os, time

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
