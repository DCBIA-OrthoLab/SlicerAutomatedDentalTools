from torch.utils.data import Dataset
import torch
import os
from AREG_IOS_utils.utils import ReadSurf, ComputeNormals, GetColorArray
from vtk.util.numpy_support import vtk_to_numpy
from AREG_IOS_utils.orientation import orientation
from AREG_IOS_utils.transformation import ScaleSurf
import glob

class TeethDatasetPatch(Dataset):
    def __init__(self,T1,T2,surf_property ):
        self.list_upper, self.list_lower = Sort(T1,T2)
        self.surf_property = surf_property
        # print(f'Init class list upper {self.list_upper}')

    def __len__(self):
            
        return len(self.list_upper)

    def __getitem__(self, args) :
        index, time = args

        surf = ReadSurf(self.list_upper[index][time])
       

        surf, matrix = orientation(surf,[[-0.5,-0.5,0],[0,0,0],[0.5,-0.5,0]],['3','8','9','14'])
        

        surf = ScaleSurf(surf)
        # print(f'sruf scale { surf}')

        surf = ComputeNormals(surf) 
        # print(f'sruf { surf}')
     
        V = torch.tensor(vtk_to_numpy(surf.GetPoints().GetData())).to(torch.float32)
        F = torch.tensor(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(torch.int64)
        CN = torch.tensor(vtk_to_numpy(GetColorArray(surf, "Normals"))/255.0,dtype=torch.float32) 


        return V, F, CN 
        
    
    def getName(self,idx):
        if isinstance(self.df,list):
            path = self.df[idx]
        else :
            path = os.path.join(self.mount_point,self.df.iloc[idx]["surf"][1:])
        name = os.path.basename(path)
        name , _ = os.path.splitext(name)

        return name
    
    def isLower(self):
        out = True
        if self.list_lower == None:
            out = False
        return out
    
    def getLowerSurf(self,idx,time):
        return ReadSurf(self.list_lower[idx][time])
    
    def getUpperSurf(self,idx,time):
        return ReadSurf(self.list_upper[idx][time])
    
    def getUpperPath(self,idx,time):
        return self.list_upper[idx][time]
    
    def getLowerPath(self,idx,time):
        return self.list_lower[idx][time]

def Sort(T1:str, T2 :str) -> tuple[list,list]:
    T1_files = glob.glob(os.path.join(T1,'*'))
    T2_files = glob.glob(os.path.join(T2,'*'))
    # print(f'in sort T1 files : {T1_files}')

    if not insideLower(T1_files) :  #check if there are Lower arches in list of file
        #if there are not lower arches
        list_reg_Upper = sort(T1_files,T2_files)
        list_reg_Lower = None

    else :  #if thehre are lower arches

        T1_Uppers = []
        T1_Lowers = []
        T2_Uppers = []
        T2_Lowers = []

        for file in T1_files :
            if isLowerUpper(file,choice='Upper'):
                T1_Uppers.append(file)
            else :
                T1_Lowers.append(file)

        for file in T2_files :
            if isLowerUpper(file,choice='Upper'):
                T2_Uppers.append(file)
            else :
                T2_Lowers.append(file)


        list_reg_Upper_tmp = sort(T1_Uppers,T2_Uppers)
        list_reg_Lower_tmp = sort(T1_Lowers,T2_Lowers)

        #organize Lower and Upper list, to have the order of file
        list_reg_Upper = []
        list_reg_Lower = []

        for Upper in list_reg_Upper_tmp :
            Upper_name = os.path.basename(Upper['T1']).replace('T1','')
            Upper_name = removeLowerUpper(Upper_name,choice='Upper')

            for Lower in list_reg_Lower_tmp :
                Lower_name = os.path.basename(Lower['T1']).replace('T1','')
                Lower_name = removeLowerUpper(Lower_name,choice='Lower')

                if Upper_name == Lower_name :
                    list_reg_Upper.append(Upper)
                    list_reg_Lower.append(Lower)
                    continue



    return list_reg_Upper, list_reg_Lower



def insideLower(list_files : list) -> bool:
    """Check if there are lower archer in list of file

    Args:
        list_files (list): contain list of file path

    Returns:
        bool: return if there are lower archer in list of file
    """
    out= False

    for file in list_files:
        if isLowerUpper(file,choice='Lower'):
            out = True
            continue
    
    return out
    

def removeLowerUpper(file_name : str, choice : str ='Upper') -> str:
    """remove the appelation in the file name of upper or lower depend of the argument choice

    Args:
        file_name (str): file path
        choice (str, optional): _description_. Defaults to 'Upper'.

    Returns:
        str:  return file name without the appelation in the file name of upper or lower depend of the argument choice
    """
    list_word = []

    if choice == "Lower":
        list_word = ['Lower','_L','L_','Mandibule','Md']
    elif choice == 'Upper':
        list_word = ['Upper','_U','U_','Maxilla','Mx']

    for word in list_word :
        file_name = file_name.replace(word,"")
    
    return file_name

def isLowerUpper(file_name : str , choice : str='Upper') -> bool:
    """ Check if the file name is for Lower of Upper depend of the choice

    Args:
        file_name (str): _description_
        choice (str, optional): _description_. Defaults to 'Upper'.

    Returns:
        bool: _description_
    """
    out = False
    list_word = []

    if choice == "Lower":
        list_word = ['Lower','_L','L_','Mandibule','Md']
    elif choice == 'Upper':
        list_word = ['Upper','_U','U_','Maxilla','Mx']

    for lower_word in list_word :
        if lower_word in file_name :
            out = True
            continue

    return out

def sort(T1_files : list ,T2_files : list) -> list[dict]:
    """Link T1 file and T2 file

    Args:
        T1_files (list): contain list of T1 files
        T2_files (list): contain list of T2 files

    Returns:
        list[dict]: exemple : [{'T1':'path/patient5T1.vtk','T2':'path/patient5T2},...,{'T1':'path/patient90T1.vtk','T2':'path/patient90T2}]
    """
    list_reg = []

    for T1_file in T1_files :
        T1_name = os.path.basename(T1_file)
        T1_name = T1_name.replace('T1','')

        for T2_file in T2_files :
            T2_name = os.path.basename(T2_file)
            T2_name = T2_name.replace('T2','')  

            if T2_name == T1_name :
                list_reg.append({'T1':T1_file,'T2':T2_file})
                continue


    return list_reg


