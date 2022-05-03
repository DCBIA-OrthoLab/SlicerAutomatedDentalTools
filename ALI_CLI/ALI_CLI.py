#!/usr/bin/env python-real

import os
import sys
import time
import json

def main(input):
    with open('env.json', 'w') as convert_file:
      convert_file.truncate(0)
      convert_file.write(input)

    print(input)
    for i in input:
        time.sleep(1)
    time.sleep(5)

if __name__ == "__main__":
    # if len (sys.argv) < 4:
    #     print("Usage: ALI_CLI <input> <sigma> <output>")
    #     sys.exit (1)
    print("ALED MATHIEU")
    main(sys.argv[1])
