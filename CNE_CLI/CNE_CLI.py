#!/usr/bin/env python-real

import sys, argparse ,shutil, os, time
print("CNE_CLI.py lancé")

def main(args):

    notesFolder_input =  args.notesFolder_input
    modelType = args.modelType
    notesType = args.notesType
    notesFolder_output = args.notesFolder_output

    print("CLI en cours de travail", flush=True)
    print(notesFolder_input, flush=True)
    print(modelType, flush=True)
    print(notesType, flush=True)
    print(notesFolder_output, flush=True)

    # print("CLI en cours de travail", flush=True)
    # sys.stdout.flush()
    # time.sleep(1)
    # print(notesFolder_input, flush=True)
    # sys.stdout.flush()
    # time.sleep(1)
    # print(modelType, flush=True)
    # sys.stdout.flush()
    # time.sleep(1)
    # print(notesType, flush=True)
    # sys.stdout.flush()
    # time.sleep(1)
    # print(notesFolder_output, flush=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('notesFolder_input',type=str)
    parser.add_argument('modelType',type=str)
    parser.add_argument("notesType",type=str)
    parser.add_argument('notesFolder_output',type=str)

    args = parser.parse_args()
    main(args)
