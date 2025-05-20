#!/usr/bin/env python-real
import argparse
import sys, os, time
import numpy as np
import SimpleITK as sitk
import json
import glob

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
            files_original = search(file_path,'.vtk','.vtp','.stl','.off','.obj','.nii.gz','.nrrd', '.mrk.json')
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

            for i in range(len(files_original['.nii.gz'])):
                files.append(files_original['.nii.gz'][i])
                
            for i in range(len(files_original['.nrrd'])):
                files.append(files_original['.nrrd'][i])
                
            for i in range(len(files_original['.mrk.json'])):
                files.append(files_original['.mrk.json'][i])

            for i in range(len(files)):
                file = files[i]

                file_pat = (os.path.basename(file)).split('_Seg')[0].split('_seg')[0].split('_Scan')[0].split('_scan')[0].split('_Or')[0].split('_OR')[0].split('_MAND')[0].split('_MD')[0].split('_MAX')[0].split('_MX')[0].split('_CB')[0].split('_lm')[0].split('_T2')[0].split('_T1')[0].split('_Cl')[0].split('.')[0]
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

            if extension ==".vtk" or extension ==".vtp" or extension ==".stl" or extension ==".off" or extension ==".obj" or extension==".nii.gz" or extension==".nrrd" or extension==".mrk.json":
                files = [file_path]
                file_pat = os.path.basename(file_path).split('_Seg')[0].split('_seg')[0].split('_Scan')[0].split('_scan')[0].split('_Or')[0].split('_OR')[0].split('_MAND')[0].split('_MD')[0].split('_MAX')[0].split('_MX')[0].split('_CB')[0].split('_lm')[0].split('_T2')[0].split('_T1')[0].split('_Cl')[0].split('.')[0].replace('.','')
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
                matrix_pat = os.path.basename(matrix).split('_Left')[0].split('_left')[0].split('_Right')[0].split('_right')[0].split('_Or')[0].split('_OR')[0].split('_MAND')[0].split('_MD')[0].split('_MAX')[0].split('_MX')[0].split('_CB')[0].split('_lm')[0].split('_T2')[0].split('_T1')[0].split('_Cl')[0].split('_MA')[0].split('_Mir')[0].split('_mir')[0].split('_Mirror')[0].split('_mirror')[0].split('.')[0].replace('.','')

                for i in range(50):
                    matrix_pat=matrix_pat.split('_T'+str(i))[0]

                if matrix_pat in patients.keys():
                    patients[matrix_pat]['matrix'].append(matrix)

        else :
            for key in patients.keys() :
                patients[key]['matrix'].append(matrix_path)

        return patients,len(files)


def ResampleImage(image, transform, reference_image, is_segmentation=False):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_segmentation else sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(transform)
        return resampler.Execute(image)

def main(patient_lineEdit, matrix_lineEdit, suffix, add_matrix_name, fromAreg, output_folder, log_path):
    
    patients, nb_files = GetPatients(patient_lineEdit, matrix_lineEdit)
    suffix_map = {
        "_CB": ("Cranial Base", "CBReg_matrix.tfm"),
        "_L": ("Maxilla", "MAXReg_matrix.tfm"),
        "_U": ("Mandible", "MANDReg_matrix.tfm"),
    }

    for key, values in patients.items():
        for scan in values['scan']:
            is_landmark = scan.endswith(".mrk.json")
            extension_scan = ''.join(Path(scan).suffixes)
            
            

            if Path(patient_lineEdit).is_dir():
                outpath = scan.replace(patient_lineEdit, output_folder)
            else:
                outpath = scan.replace(os.path.dirname(patient_lineEdit), output_folder)

            if not os.path.exists(os.path.dirname(outpath)):
                os.makedirs(os.path.dirname(outpath))
                

            # Find matrix
            matrix_candidates = []
            if fromAreg == "True" and is_landmark:
                matched = False
                for suffix, (subdir, matrix_filename) in suffix_map.items():
                    if suffix in os.path.basename(scan):
                        patient_id = os.path.basename(scan).split('_')[0]
                        matrix_path = os.path.join(
                            matrix_lineEdit,
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
                tfm = sitk.ReadTransform(matrix)
                matrix_suffix = f"_{Path(matrix).stem}" if add_matrix_name == "True" else ""
                
                image = None if is_landmark else sitk.ReadImage(scan)

                if is_landmark:
                    with open(scan, 'r') as f:
                        lm_data = json.load(f)

                    points = np.array([p['position'] + [1.0] for p in lm_data['markups'][0]['controlPoints']]).T

                    if isinstance(tfm, sitk.AffineTransform):
                        mat = np.eye(4)
                        mat[:3, :3] = np.array(tfm.GetMatrix()).reshape(3, 3)
                        mat[:3, 3] = np.array(tfm.GetTranslation())
                        transformed = (mat @ points).T[:, :3]
                        for i, p in enumerate(lm_data['markups'][0]['controlPoints']):
                            p['position'] = transformed[i].tolist()

                        out_file = outpath.replace(".mrk.json", f"{suffix}{matrix_suffix}.mrk.json")
                        with open(out_file, 'w') as f:
                            json.dump(lm_data, f, indent=2)
                    else:
                        print(f"WARNING: Landmark transform is not affine. Skipping {matrix}")
                    continue

                # For volumes
                try:
                    if isinstance(tfm, sitk.CompositeTransform):
                        # Require reference image with matching shape
                        ref_guess = matrix.replace("_transform.tfm", ".nii.gz")
                        if os.path.exists(ref_guess):
                            reference_image = sitk.ReadImage(ref_guess)
                        else:
                            print(f"WARNING: CompositeTransform but no reference image found at {ref_guess}. Using scan as fallback.")
                            reference_image = image
                    else:
                        reference_image = image

                    output_image = ResampleImage(image, tfm, reference_image)
                    out_file = outpath.replace(extension_scan, f"{suffix}{matrix_suffix}{extension_scan}")
                    sitk.WriteImage(output_image, out_file)
                    
                        
                except Exception as e:
                    print(f"ERROR processing {scan} with matrix {matrix}: {e}")
                    continue
        
            with open(log_path, "a") as log_f:
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
    parser.add_argument("suffix", type=str)
    parser.add_argument("matrix_name", type=str)
    parser.add_argument("fromAreg", type=str)
    parser.add_argument("output_folder", type=str)
    parser.add_argument("log_path", type=str)    

    args = parser.parse_args()

    main(args.input_patient, args.input_matrix, args.suffix, args.matrix_name, args.fromAreg, args.output_folder, args.log_path)
