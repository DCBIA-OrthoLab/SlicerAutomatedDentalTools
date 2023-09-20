from pathlib import Path
import os
import glob

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
            files_original = search(file_path,'.vtk','.vtp','.stl','.off','.obj','.nii.gz')
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

            if extension ==".vtk" or extension ==".vtp" or extension ==".stl" or extension ==".off" or extension ==".obj" or extension==".nii.gz" :
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
                matrix_pat = os.path.basename(matrix).split('_Left')[0].split('_left')[0].split('_Right')[0].split('_right')[0].split('_T1')[0].split('_T2')[0].split('_MA')[0]

                for i in range(50):
                    matrix_pat=matrix_pat.split('_T'+str(i))[0]

                if matrix_pat in patients.keys():
                    patients[matrix_pat]['matrix'].append(matrix)

        else : 
            for key in patients.keys() :
                patients[key]['matrix'].append(matrix_path)

        return patients,len(files)