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
    
    #     with open(log_path, "a") as log_f:
    #         log_f.write(str(1))
    #
    #     print(f"""<filter-progress>{0}</filter-progress>""")
    #     sys.stdout.flush()
    #     time.sleep(0.2)
    #     print(f"""<filter-progress>{2}</filter-progress>""")
    #     sys.stdout.flush()
    #     time.sleep(0.2)
    #     print(f"""<filter-progress>{0}</filter-progress>""")
    #     sys.stdout.flush()
    #     time.sleep(0.2)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("summary_folder", type=str)
    parser.add_argument("output_folder", type=str)
    parser.add_argument("log_path", type=str)

    args = parser.parse_args()

    main(args)
