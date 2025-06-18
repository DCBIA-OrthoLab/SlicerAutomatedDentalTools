#!/usr/bin/env python-real

import sys
import os
import time
import argparse
import SimpleITK as sitk

fpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(fpath)

from ASO_CBCT_utils import (
    ICP,
    ExtractFilesFromFolder,
    MergeJson,
    WriteJson,
    GetPatients,
)


def main(args):

    scan_extension = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
    lm_extension = [".json"]

    list_landmark = args.list_landmark[0].split(" ")
    input_dir, gold_dir, out_dir = (
        args.input[0],
        args.gold_folder[0],
        args.output_folder[0],
    )

    MergeJson(input_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    patients = GetPatients(input_dir)

    gold_file, gold_json_file = ExtractFilesFromFolder(
        gold_dir, scan_extension, lm_extension, gold=True
    )
    if not os.path.exists(os.path.split(args.log_path)[0]):
        os.mkdir(os.path.split(args.log_path)[0])

    with open(args.log_path, "w") as log_f:
        log_f.truncate(0)

    idx=0
    for patient, data in patients.items():

        with open(args.log_path, "r+") as log_f:
            log_f.write(str(idx+1))

        try:
            input_file, input_json_file, input_transform = data["scan"], data["json"], data["tfm"]

            output, source_transformed, TransformSITK = ICP(
                input_file, input_json_file, gold_file, gold_json_file, list_landmark, input_transform
            )

            if output is None:
                print("ICP failed for patient:", patient)
                continue

            # Write JSON
            dir_json = os.path.dirname(input_json_file.replace(input_dir, out_dir))
            if not os.path.exists(dir_json):
                os.makedirs(dir_json)
            json_path = os.path.join(
                dir_json, patient + "_lm_" + args.add_inname[0] + ".mrk.json"
            )

            if not os.path.exists(json_path):
                WriteJson(source_transformed, json_path)

            # Write Scan
            dir_scan = os.path.dirname(input_file.replace(input_dir, out_dir))
            if not os.path.exists(dir_scan):
                os.makedirs(dir_scan)

            file_outpath = os.path.join(
                dir_scan, patient + "_" + args.add_inname[0] + ".nii.gz"
            )
            if not os.path.exists(file_outpath):
                sitk.WriteImage(output, file_outpath)
                
            transform_outpath = os.path.join(
                dir_scan, patient + "_" + args.add_inname[0] + "_transform.tfm"
            )
            if not os.path.exists(transform_outpath):
                sitk.WriteTransform(TransformSITK, transform_outpath)

        except KeyError:
            print("patient {} does not have scan and/or json file(s)".format(patient))

        print(f"""<filter-progress>{0}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.2)
        print(f"""<filter-progress>{2}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.2)
        print(f"""<filter-progress>{0}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.2)


if __name__ == "__main__":

    print("Starting")
    print(sys.argv)

    parser = argparse.ArgumentParser()

    parser.add_argument("input", nargs=1)
    parser.add_argument("gold_folder", nargs=1)
    parser.add_argument("output_folder", nargs=1)
    parser.add_argument("add_inname", nargs=1)
    parser.add_argument("list_landmark", nargs=1)
    parser.add_argument("log_path", type=str)

    args = parser.parse_args()

    main(args)
