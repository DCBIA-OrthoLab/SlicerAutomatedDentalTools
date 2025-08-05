#!/usr/bin/env python-real
import argparse
import json
import glob
import sys, os, time
import SimpleITK as sitk

from pathlib import Path

fpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(fpath)


def search(path, *args):
        """
        Return a dictionary with args element as key and a list of file in path directory finishing by args extension for each key

        Example:
        args = ('json',['.nii.gz','.nrrd'])
        return:
            {
                'json' : ['path/a.json', 'path/b.json','path/c.json'],
                '.nii.gz' : ['path/a.nii.gz', 'path/b.nii.gz']
                '.nrrd.gz' : ['path/c.nrrd']
            }
        """
        arguments = []
        for arg in args:
            if type(arg) == list:
                arguments.extend(arg)
            else:
                arguments.append(arg)
        return {
            key: [
                i
                for i in glob.iglob(
                    os.path.normpath("/".join([path, "**", "*"])), recursive=True
                )
                if i.endswith(key)
            ]
            for key in arguments
        }


def GetPatients(file_path:str,matrix_path:str):
        """
            Return a dictionnary with the patients names for the key. Their .nii.gz/.vtk/.vtp/.stl./.off files and their matrix.
            exemple :
            input : file_path matrix_path
            output :
            ('patient1': {'scan':[file_path_1_patient1.nii.gz,file_path_2_patient1.vtk],'matrix':[matrix_path_1_patient1.tfm,matrix_path_2_patient1.h5]})
        """
        patients = {}
        files = []

        if Path(file_path).is_dir():
            files_original = search(file_path,'.vtk','.vtp','.stl','.off','.obj','.nii', '.nii.gz','.nrrd', '.mrk.json')
            files = []
            for i in range(len(files_original['.vtk'])):
                files.append(files_original['.vtk'][i])

            for i in range(len(files_original['.vtp'])):
                files.append(files_original['.vtp'][i])

            for i in range(len(files_original['.stl'])):
                files.append(files_original['.stl'][i])

            for i in range(len(files_original['.off'])):
                files.append(files_original['.off'][i])

            for i in range(len(files_original['.obj'])):
                files.append(files_original['.obj'][i])
                
            for i in range(len(files_original['.nii'])):
                files.append(files_original['.nii'][i])

            for i in range(len(files_original['.nii.gz'])):
                files.append(files_original['.nii.gz'][i])
                
            for i in range(len(files_original['.nrrd'])):
                files.append(files_original['.nrrd'][i])
                
            for i in range(len(files_original['.mrk.json'])):
                files.append(files_original['.mrk.json'][i])

            for i in range(len(files)):
                file = files[i]

                file_pat = (os.path.basename(file)).split('_Seg')[0].split('_seg')[0].split('_Scan')[0].split('_scan')[0].split('_Or')[0].split('_OR')[0].split('_MAND')[0].split('_MD')[0].split('_MAX')[0].split('_MX')[0].split('_CB')[0].split('_lm')[0].split('_T2')[0].split('_T1')[0].split('_Cl')[0].split('_MR')[0].split('.')[0]
                for i in range(50):
                    file_pat=file_pat.split('_T'+str(i))[0]

                if file_pat not in patients.keys():
                    patients[file_pat] = {}
                    patients[file_pat]['scan'] = []
                    patients[file_pat]['matrix'] = []
                patients[file_pat]['scan'].append(file)

        else :
            fname, extension = os.path.splitext(file_path)

            try :
                fname, extension2 = os.path.splitext(os.path.basename(fname))
                extension = extension2+extension
            except :
                print("not a .nii.gz")

            if extension ==".vtk" or extension ==".vtp" or extension ==".stl" or extension ==".off" or extension ==".obj" or extension==".nii" or extension==".nii.gz" or extension==".nrrd" or extension==".mrk.json":
                files = [file_path]
                file_pat = os.path.basename(file_path).split('_Seg')[0].split('_seg')[0].split('_Scan')[0].split('_scan')[0].split('_Or')[0].split('_OR')[0].split('_MAND')[0].split('_MD')[0].split('_MAX')[0].split('_MX')[0].split('_CB')[0].split('_lm')[0].split('_T2')[0].split('_T1')[0].split('_Cl')[0].split('_MR')[0].split('.')[0].replace('.','')
                for i in range(50):
                    file_pat=file_pat.split('_T'+str(i))[0]

                if file_pat not in patients.keys():
                    patients[file_pat] = {}
                    patients[file_pat]['scan'] = []
                    patients[file_pat]['matrix'] = []
                patients[file_pat]['scan'].append(file_path)


        if Path(matrix_path).is_dir():
            matrixes_original = search(matrix_path,'.npy','.h5','.tfm','.mat','.txt')
            matrixes = []

            for i in range(len(matrixes_original['.npy'])):
                matrixes.append(matrixes_original['.npy'][i])

            for i in range(len(matrixes_original['.h5'])):
                matrixes.append(matrixes_original['.h5'][i])

            for i in range(len(matrixes_original['.tfm'])):
                matrixes.append(matrixes_original['.tfm'][i])

            for i in range(len(matrixes_original['.mat'])):
                matrixes.append(matrixes_original['.mat'][i])

            for i in range(len(matrixes_original['.txt'])):
                matrixes.append(matrixes_original['.txt'][i])

            for i in range(len(matrixes)):
                matrix = matrixes[i]
                matrix_pat = os.path.basename(matrix).split('_Left')[0].split('_left')[0].split('_Right')[0].split('_right')[0].split('_Or')[0].split('_OR')[0].split('_MAND')[0].split('_MD')[0].split('_MAX')[0].split('_MX')[0].split('_CB')[0].split('_lm')[0].split('_T2')[0].split('_T1')[0].split('_Cl')[0].split('_MA')[0].split('_Mir')[0].split('_mir')[0].split('_Mirror')[0].split('_mirror')[0].split('_MR')[0].split('.')[0].replace('.','')

                for i in range(50):
                    matrix_pat=matrix_pat.split('_T'+str(i))[0]

                if matrix_pat in patients.keys():
                    patients[matrix_pat]['matrix'].append(matrix)

        else :
            for key in patients.keys() :
                patients[key]['matrix'].append(matrix_path)

        return patients,len(files)

def apply_transform_to_landmarks(scan_path, transform, output_path):
    with open(scan_path, 'r') as f:
        lm_data = json.load(f)

    try:
        tfm_inverted = transform.GetInverse()
    except RuntimeError:
        print(f"WARNING: Could not invert transform for {scan_path}. Skipping.")
        return

    for point in lm_data['markups'][0]['controlPoints']:
        point['position'] = list(tfm_inverted.TransformPoint(point['position']))

    with open(output_path, 'w') as f:
        json.dump(lm_data, f, indent=2)
        

def apply_transform_to_image(image, transform, reference, output_path, scan_path, is_seg=False):
    if isinstance(transform, sitk.CompositeTransform):
        ref_guess_gz = scan_path.replace("_transform.tfm", ".nii.gz")
        ref_guess_nii = scan_path.replace("_transform.tfm", ".nii")

        if os.path.exists(ref_guess_gz):
            reference = sitk.ReadImage(ref_guess_gz)
        elif os.path.exists(ref_guess_nii):
            reference = sitk.ReadImage(ref_guess_nii)
        else:
            print(f"WARNING: CompositeTransform but no reference found at {ref_guess_gz} or {ref_guess_nii}. Using image as fallback.")
            reference = image

    resampled_image = ResampleImage(image, transform, reference, is_seg)
    sitk.WriteImage(resampled_image, output_path)

def ResampleImage(image, transform, reference=None, is_seg=False):
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_seg else sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)

    if reference is not None:
        resampler.SetReferenceImage(reference)
    else:
        # mimic the behavior of ResampleScalarVectorDWIVolume without reference
        resampler.SetSize(image.GetSize())
        resampler.SetOutputSpacing(image.GetSpacing())
        resampler.SetOutputDirection(image.GetDirection())
        new_origin = transform.TransformPoint(image.GetOrigin())
        resampler.SetOutputOrigin(new_origin)

    return resampler.Execute(image)

def main(args):
    patients, _ = GetPatients(args.input_patient, args.input_matrix)
    suffix_map = {
        "_CB": ("Cranial Base", "CBReg_matrix.tfm"),
        "_L": ("Maxilla", "MAXReg_matrix.tfm"),
        "_U": ("Mandible", "MANDReg_matrix.tfm"),
    }
    
    is_seg = args.is_seg.lower() == "true"
    
    reference_image = None
    if (args.reference_file != "None") and Path(args.reference_file).exists():
        try:
            reference_image = sitk.ReadImage(args.reference_file)
        except Exception as e:
            print(f"WARNING: Could not read reference image: {e}")
            reference_image = None
    else:
        print(f"INFO: No valid reference image provided. Will use each scan's geometry.")

    for key, values in patients.items():
        for scan in values['scan']:
            is_landmark = scan.endswith(".mrk.json")
            extension_scan = ''.join(Path(scan).suffixes)
            
            if Path(args.input_patient).is_dir():
                outpath = scan.replace(args.input_patient, args.output_folder)
            else:
                outpath = scan.replace(os.path.dirname(args.input_patient), args.output_folder)

            os.makedirs(os.path.dirname(outpath), exist_ok=True)
                
            # Find matrix
            matrix_candidates = []
            if args.fromAreg == "True" and is_landmark:
                matched = False
                for suffix, (subdir, matrix_filename) in suffix_map.items():
                    if suffix in os.path.basename(scan):
                        patient_id = os.path.basename(scan).split('_')[0]
                        matrix_path = os.path.join(
                            args.matrix_lineEdit,
                            subdir,
                            f"{patient_id}_OutReg",
                            f"{patient_id}_{matrix_filename}"
                        )
                        if os.path.exists(matrix_path):
                            matrix_candidates = [matrix_path]
                            matched = True
                        else:
                            print(f"WARNING: Matrix not found for {scan} at {matrix_path}")
                        break
                    
                if not matched:
                    print(f"WARNING: No suffix match for {scan}")
                    continue
            else:
                matrix_candidates = values['matrix']

            for matrix in matrix_candidates:
                try:
                    tfm = sitk.ReadTransform(matrix)
                except Exception as e:
                    print(f"ERROR reading transform {matrix}: {e}")
                    continue

                matrix_suffix = f"_{Path(matrix).stem}" if args.matrix_name == "True" else ""
                out_suffix = f"{args.suffix}{matrix_suffix}"
                
                # For landmarks
                if is_landmark:
                    out_file = outpath.split(".mrk.json")[0] + out_suffix + ".mrk.json"
                    apply_transform_to_landmarks(scan, tfm, out_file)
                    continue

                # For volumes
                try:
                    image = sitk.ReadImage(scan)
                    if "mirror" in os.path.basename(matrix).lower():
                        tfm = sitk.AffineTransform(tfm)
                        # Center the transform around image center
                        center = image.TransformContinuousIndexToPhysicalPoint([
                            (sz - 1) / 2.0 for sz in image.GetSize()
                        ])
                        tfm.SetCenter(center)
                        
                        local_reference = image
                    else:
                        local_reference = reference_image if reference_image is not None else image
                    
                    out_file = outpath.split(extension_scan)[0] + out_suffix + extension_scan
                    apply_transform_to_image(image, tfm, local_reference, out_file, matrix, is_seg=is_seg)
                except Exception as e:
                    print(f"ERROR processing {scan} with matrix {matrix}: {e}")
                    continue
        
            with open(args.log_path, "a") as log_f:
                log_f.write(str(1))
                            
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

    parser = argparse.ArgumentParser()

    parser.add_argument("input_patient", type=str)
    parser.add_argument("input_matrix", type=str)
    parser.add_argument("reference_file", type=str)
    parser.add_argument("suffix", type=str)
    parser.add_argument("matrix_name", type=str)
    parser.add_argument("fromAreg", type=str)
    parser.add_argument("output_folder", type=str)
    parser.add_argument("log_path", type=str)
    parser.add_argument("is_seg", type=str)

    args = parser.parse_args()

    main(args)
