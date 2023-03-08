from dataclasses import dataclass,field ,astuple, asdict
from typing import Tuple, Union, List
import os
import glob
from itertools import chain


@dataclass(init=True)
class Upper:
    name1 : str = field(repr=False ,default='Upper')
    name2 : str = field(repr=False,default= '_U_' )

    def __str__(self) -> str:
        return 'Upper'

    def __eq__(self, __o: object) -> bool:
        out = False
        if isinstance(__o,Upper) :
            out = True

        elif isinstance(__o,str):
            if __o == 'Upper':
                out = True 
        return out


    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)



@dataclass(init=True)
class Lower :
    name1 : str = field(repr=False, default='Lower')
    name2: str = field(repr=False, default='_L_')


    def __str__(self) -> str:
        return 'Lower'

    def __eq__(self, __o: object) -> bool:
        out = False
        if isinstance(__o,Lower) :
            out = True

        elif isinstance(__o,str):
            if __o == 'Lower':
                out = True 
        return out


    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)


@dataclass(init=True,repr=True)
class Jaw:
    upper : Upper = field(init = False, repr=False,default=Upper())
    lower : Lower = field(init=False, repr=False, default=Lower())
    actual : Union[Upper, Lower]

    def __init__(self,actual) -> None:
        assert isinstance(actual,(Upper,Lower,str))
        if isinstance(actual,str):
            actual = Files.TypeOfJaw(actual)
        self.actual = actual


    def inv(self):
        out =  self.upper
        if isinstance(self.actual,Upper):
            out = self.lower
        
    
        return str(out)



    def __str__(self) -> str:
        return str(self.actual)

    def __eq__(self,other):
        out = False
        if isinstance(other.actual,self.actual):
            out = True
        return out

    def __call__(self):

        return str(self.actual)







@dataclass(init=True,repr=True,eq=True,frozen=True)
class Jaw_File :
    
    vtk : str
    jaw : Jaw
    name : str
    json : Union[str , None] = field(default=None)




@dataclass(init=True,repr=True,eq=True)
class Mouth_File:
    Upper : Union[Jaw_File , str]
    Lower : Union [Jaw_File, str]
    name : str





class Files:

    def __init__(self,folder : str) -> None:

        self.list_file : List[Union[Mouth_File  , Jaw_File]] = []
        self.folder : str = folder
        self.extension = ['.vtk','.vtp','.stl','.off','.obj']




    def __name_file__(self,name_file : str):
        name_file = os.path.basename(name_file)
        name_file, _ = os.path.splitext(name_file)
        jaw = self.TypeOfJaw(name_file)
        name_file = self.__remove_jaw__(name_file,jaw)
        if '_out' in name_file :
            name_file = name_file.replace('_out','').replace('Or','')

        
        return jaw, name_file


    def __remove_jaw__(self,name_file : str,jaw : Union[Upper,Lower]):
        work = False
        for st in astuple(jaw):
            if st.lower() in name_file.lower():
                index = name_file.lower().find(st.lower())
                name_file = name_file[:index]+name_file[index+len(st):]
                work = True
        
        if work :
            self.__remove_jaw__(name_file,jaw)

        return name_file



    @staticmethod
    def TypeOfJaw(name_file : str):
        out = None

        if True in [upper.lower() in name_file.lower() for upper in astuple(Upper())]:
            out =Upper()

        elif True in [upper.lower() in name_file.lower() for upper in astuple(Lower())]:
            out = Lower()

        if out is None:
            raise ValueError(f"dont found the jaw's type to {name_file}")
        return out


        




    def __len__(self):
        return len(self.list_file)

    def __iter__(self):
        self.iter=-1
        return self


    def __next__(self):
        self.iter += 1 
        if self.iter>= len(self.list_file):
            raise StopIteration

        
        
        return asdict(self.list_file[self.iter])



    def search(self,path,*args):
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
        arguments=[]
        for arg in args:
            if type(arg) == list:
                arguments.extend(arg)
            else:
                arguments.append(arg)
        out = {key: [i for i in glob.iglob(os.path.normpath("/".join([path,'**','*'])),recursive=True) if i.endswith(key)] for key in arguments}

        for key , values in out.items():
            lst = []
            for value in values :
                if os.path.isfile(value):
                    lst.append(value)
            out[key]=lst

        return out 



class Files_vtk_link(Files):
    """
    From folder path,find lower upper jaw belong to the same patient

    So, list_file get Mouth file. In Mouth file there are upper lower jaw and name of the pattient

    Args:
        Files (_type_): _description_
    """
    def __init__(self, folder: str) -> None:
        super().__init__(folder)
        self.organise(folder)
        
    def organise(self,folder):
        list_vtk =list(chain.from_iterable(self.search(folder,self.extension).values()))
        

        dic ={}
        for vtk in list_vtk:
            jaw , name = self.__name_file__(vtk)
            if name in dic:
                dic[name].append(vtk)
            else :
                dic[name]= [vtk]


        for key , value in dic.items():
            if len(value)==2:
                vtk1 = value[0]
                vtk2 = value[1]
                jaw1, name1 = self.__name_file__(vtk1)
                if isinstance(jaw1,Lower):
                    vtk1, vtk2 = vtk2, vtk1

                self.list_file.append(Mouth_File(vtk1,vtk2,name1))

        return self.list_file


# @dataclass()
# class Files_vtk_json_link2(Files):
#     jaw : str = field(repr=False)

#     def __post_init__(self):
#         self.list_file = []
#         self.list_file = self.__organise__(self.folder)

#     def __organise__(self,folder):
#         list_file = []
#         dic = self.search(folder,'.vtk','.json')
#         list_json = dic['.json']
#         list_vtk = dic['.vtk']
#         list_notgoodjaw = []
#         for vtk in list_vtk:
#             if self.jaw != self.TypeOfJaw(vtk):
#                 list_notgoodjaw.append(vtk)


#         list_vtk = list(set(list_vtk)-set(list_notgoodjaw))
#         list_jaw = []
#         json_remove = None
#         for vtk in list_vtk :
#             vtk_jaw , vtk_name = self.__name_file__(vtk)
#             for json in list_json :
#                 json_jaw , json_name = self.__name_file__(json)
#                 if vtk_name in json_name :
#                     fil = Jaw_File(json= json , vtk= vtk, jaw = json_jaw, name = vtk_name)
#                     list_jaw.append(fil)
#                     json_remove = json
#                     break
#             if json_remove is not None :
#                 list_json.remove(json)

#             json_remove = None



        # for jaw_f in list_jaw :
        #     for vtk in list_notgoodjaw :
        #         vtk_jaw , vtk_name = self.__name_file__(vtk)
        #         if vtk_name == jaw_f.name :
        #             if self.jaw == 'Upper':
        #                 list_file.append(Mouth_File(Upper=jaw_f, Lower= vtk, name = vtk_name))
        #             else :
        #                 list_file.append(Mouth_File(Upper = vtk , Lower= jaw_f, name = vtk_name))


        # return list_file



class Files_vtk_json(Files):
    """
    From path folder, find landmark(json) and jaw(vtk) matche together.
    So, in list_files there are Jaw_file with landmark(json), jaw(vtk), lower/upper and name of patient 
    There is only one landmark by Jaw_file
    Args:
        Files (_type_): _description_
    """
    def __init__(self, folder: str) -> None:
        super().__init__(folder)
        self.list_file = self.organise(folder)
        


    def organise(self,folder):
        list_file= []
        dic =self.search(folder,self.extension,'.json')

        list_json = dic['.json']
        list_json.append('Upper_nioegfjhdfjkdffdhjmndfhnmdfhj')
        list_vtk = list(chain.from_iterable(map(dic.get,self.extension)))
        json_remove = None
        for vtk in list_vtk:
            vtk_jaw , vtk_name = self.__name_file__(vtk)
            for json in list_json:
                json_jaw , json_name = self.__name_file__(json)
                if vtk_name in json_name and vtk_jaw==json_jaw :
                    fil = Jaw_File(json = json,vtk= vtk, jaw = json_jaw,name = vtk_name)
                    list_file.append(fil)
                    json_remove = json
                    break

            if json_remove is not None :
                list_json.remove(json_remove)

            json_remove = None

        return list_file



class Files_vtk_json_link(Files_vtk_json):
    """
    From  folder, match files belong to the same patient: upper jaw(vtk) , upper landmark(json),  lower jaw(vtk) and lower landmark(json).
    So, in list_files there are Mouth_file with 2 Jaw_file and name of patient. Each Jaw_file contain landmark file, jaw file and upper or lower.
    Only one json file is taken by jaw 
    

    Args:
        Files_vtk_json (_type_): _description_
    """
    def __init__(self, folder: str) -> None:
        super().__init__(folder)
        self.organise(folder)


    def organise(self,folder):
        list_file = super().organise(folder)
        list_upper = []
        list_lower = []
        for fil in list_file:
            if isinstance(fil.jaw,Upper):
                list_upper.append(fil)
            else:
                list_lower.append(fil)

        lower_remove = None
        for upper in list_upper:
            for lower in list_lower:
                if upper.name == lower.name :
                    fil = Mouth_File(upper,lower,upper.name)
                    self.list_file.append(fil)
                    lower_remove = lower
                    break
            if lower_remove is not None:
                list_lower.remove(lower_remove)

            lower_remove =None

        return self.list_file




class Files_vtk_json_semilink(Files):
    """
    From  folder, match files belong to the same patient: upper jaw(vtk) , upper landmark(json),  lower jaw(vtk) and lower landmark(json). the json files are not required to use this class unlike to Files_vtk_json_link
    So, in list_files there are Mouth_file with 2 Jaw_file and name of patient. Each Jaw_file contain landmark file, jaw file and upper or lower.
    Only one json file is taken by jaw 
    

    Args:
        Files (_type_): _description_
    """
    def __init__(self, folder: str) -> None:
        super().__init__(folder)
        self.list_file = self.organise(folder)


    def organise(self,folder):
        list_file = []
        dic = self.search(folder,self.extension,'.json')
        list_vtk = list(chain.from_iterable(map(dic.get,self.extension)))
        list_json = dic['.json']

        fil = {'Upper':[],'Lower':[]}
        json_remove = None
        for vtk in list_vtk :
            vtk_jaw , vtk_name = self.__name_file__(vtk)
            
            for json in list_json :
                json_jaw , json_name = self.__name_file__(json)

                if vtk_name in json_name and vtk_jaw == json_jaw:
                    fil[str(vtk_jaw)].append(Jaw_File(json=json, vtk=vtk , name=vtk_name, jaw = json_jaw))
                    json_remove = json

                    break

                
            if json_remove is not None :
                list_json.remove(json_remove)

            else :
                fil[str(vtk_jaw)].append(Jaw_File(vtk=vtk,name=vtk_name, jaw = vtk_jaw))

            json_remove = None



        for upper in fil['Upper']:
            for lower in fil['Lower']:
                if upper.name == lower.name :
                    list_file.append(Mouth_File(Upper=upper,Lower=lower, name=upper.name))
                    break


        return list_file


            


            





