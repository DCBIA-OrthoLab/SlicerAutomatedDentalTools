import numpy as np
import os,json
import glob


def Search(path : str,*args ) :
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

    Input : Path of the folder/file, list of the type (str) of file we need 
    Output : dictionnary with the key and the associated path
    """
    
    arguments=[]
    
    for arg in args:
        if type(arg) == list:
            arguments.extend(arg)
            
        else:
            arguments.append(arg)
            
    #result = {key: [i for i in glob.iglob(os.path.join(path,'**','*'),recursive=True),if i.endswith(key)] for key in arguments}
    
    result = {}  # Initialize an empty dictionary
    
    for key in arguments:

        files_matching_key = [] # empty list 'files_matching_key' to store the file paths that end with the current 'key'

        if os.path.isdir(path):
            # Use 'glob.iglob' to find all file paths ending with the current 'key' in the 'path' directory
            # and store the generator object returned by 'glob.iglob' in a variable 'files_generator'
            
            files_list = glob.iglob(os.path.join(path,'**', '*'),recursive=True)
            for i in files_list:
        
                if i.endswith(key):
                    # If the file path ends with the current 'key', append it to the 'files_matching_key' list
                    files_matching_key.append(i)
                
                

        else :  # if a file is choosen
            if path.endswith(key) :
                files_matching_key.append(path)
        
        # Assign the resulting list to the 'key' in the 'result' dictionary
        result[key] = files_matching_key

    return result
    