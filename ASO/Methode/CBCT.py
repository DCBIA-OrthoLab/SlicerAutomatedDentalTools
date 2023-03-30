from Methode.Methode import Methode
from Methode.Progress import DisplayASOCBCT, DisplayALICBCT
import webbrowser
import os 
import slicer
import json
import time
import qt

class CBCT(Methode):
    def __init__(self, widget):
        super().__init__(widget)

    def NumberScan(self, scan_folder: str):
        scan_extension = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
        dic = super().search(scan_folder,scan_extension)
        lenscan=0
        for key in scan_extension:
            lenscan+=len(dic[key])
        return lenscan
        
    def PatientScanLandmark(self, dic, scan_extension, lm_extension):
        patients = {}

        for extension,files in dic.items():
            for file in files:
                file_name = os.path.basename(file).split(".")[0]
                patient = file_name.split('_scan')[0].split('_Scanreg')[0].split('_lm')[0]

                if patient not in patients.keys():
                    patients[patient] = {"dir": os.path.dirname(file),"lmrk":[]}
                if extension in scan_extension:
                    patients[patient]["scan"] = file
                if extension in lm_extension:
                    patients[patient]["lmrk"].append(file)

        return patients
    
    def getReferenceList(self):
        return {
            "Occlusal and Midsagittal Plane": "https://github.com/lucanchling/ASO_CBCT/releases/download/v01_goldmodels/Occlusal_Midsagittal_Plane.zip",
            "Frankfurt Horizontal and Midsagittal Plane": "https://github.com/lucanchling/ASO_CBCT/releases/download/v01_goldmodels/Frankfurt_Horizontal_Midsagittal_Plane.zip",

        }

    def TestReference(self, ref_folder: str):
        out = None
        scan_extension = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
        lm_extension = [".json"]

        if self.NumberScan(ref_folder) == 0 :
            out = 'The selected folder must contain scans'

        if self.NumberScan(ref_folder) > 1 :
            out = 'The selected folder must contain only 1 case'
        
        return out

    def TestCheckbox(self,dic_checkbox):
        list_landmark = self.CheckboxisChecked(dic_checkbox)
        out = None
        if len(list_landmark) < 3:
             out = 'Please select at least 3 landmarks\n'
        return out    

    def TestModel(self, model_folder: str,lineEditName) -> str:

        if lineEditName == 'lineEditModelSegOr':
            if len(super().search(model_folder,'ckpt')['ckpt']) == 0:
                return 'Folder must have Pre ASO models files'
            else:
                return None
        
        if lineEditName == 'lineEditModelAli':
            if len(super().search(model_folder,'pth')['pth']) == 0:
                return 'Folder must have ALI models files'
            else:
                return None

    def TestProcess(self, **kwargs) -> str:
        out=''

                
        testcheckbox = self.TestCheckbox(kwargs['dic_checkbox'])
        if testcheckbox is not None:
            out+=testcheckbox

        if kwargs['input_folder'] == '':
            out+= 'Please select an input folder\n'

        if kwargs['gold_folder'] == '':
            out+= 'Please select a reference folder\n'

        if kwargs['folder_output'] == '':
            out+= 'Please select an output folder\n'

        if kwargs['add_in_namefile']== '':
            out += 'Please select an extension for output files\n'

        if out == '':
            out = None

        return out

    def getSegOrModelList(self):
        return ("PreASOModels", "https://github.com/lucanchling/ASO_CBCT/releases/download/v01_preASOmodels/PreASOModels.zip")

    def getALIModelList(self):
        return ("ALIModels", "https://github.com/lucanchling/ALI_CBCT/releases/download/models_v01/")

    def DicLandmark(self):
        dic = {'Landmark':
                {'Cranial Base' : sorted(['Ba', 'S', 'N', 'RPo', 'LPo', 'RFZyg', 'LFZyg', 'C2', 'C3', 'C4']),

                'Upper' : sorted(['RInfOr', 'LInfOr', 'LMZyg', 'RPF', 'LPF', 'PNS', 'ANS', 'A', 'UR3O', 'UR1O', 'UL3O', 'UR6DB', 'UR6MB', 'UL6MB', 'UL6DB', 'IF', 'ROr', 'LOr', 'RMZyg', 'RNC', 'LNC', 'UR7O', 'UR5O', 'UR4O', 'UR2O', 'UL1O', 'UL2O', 'UL4O', 'UL5O', 'UL7O', 'UL7R', 'UL5R', 'UL4R', 'UL2R', 'UL1R', 'UR2R', 'UR4R', 'UR5R', 'UR7R', 'UR6MP', 'UL6MP', 'UL6R', 'UR6R', 'UR6O', 'UL6O', 'UL3R', 'UR3R', 'UR1R']),

                'Lower' : sorted(['RCo', 'RGo', 'Me', 'Gn', 'Pog', 'PogL', 'B', 'LGo', 'LCo', 'LR1O', 'LL6MB', 'LL6DB', 'LR6MB', 'LR6DB', 'LAF', 'LAE', 'RAF', 'RAE', 'LMCo', 'LLCo', 'RMCo', 'RLCo', 'RMeF', 'LMeF', 'RSig', 'RPRa', 'RARa', 'LSig', 'LARa', 'LPRa', 'LR7R', 'LR5R', 'LR4R', 'LR3R', 'LL3R', 'LL4R', 'LL5R', 'LL7R', 'LL7O', 'LL5O', 'LL4O', 'LL3O', 'LL2O', 'LL1O', 'LR2O', 'LR3O', 'LR4O', 'LR5O', 'LR7O', 'LL6R', 'LR6R', 'LL6O', 'LR6O', 'LR1R', 'LL1R', 'LL2R', 'LR2R']),

                }}

        return dic

        

        
    def Suggest(self):
        return ['Ba','S','N','RPo','LPo','ROr','LOr']


    def CheckboxisChecked(self,diccheckbox : dict, in_str = False):
        out=''
        listchecked = []
        if not len(diccheckbox) == 0:
            for checkboxs in diccheckbox.values():
                for checkbox in checkboxs:
                    if checkbox.isChecked():
                        listchecked.append(checkbox.text)
        if in_str:
            listchecked_str = ''
            for i,lm in enumerate(listchecked):
                if i<len(listchecked)-1:
                    listchecked_str+= lm+' '
                else:
                    listchecked_str+=lm
            return listchecked_str
    
        return listchecked
    

class Semi_CBCT(CBCT):
    

    def getTestFileList(self):
        return ("Semi-Automated", "https://github.com/lucanchling/ASO_CBCT/releases/download/TestFiles/Occlusal_Midsagittal_Test.zip")

    def TestScan(self, scan_folder: str):
        out = ''
        scan_extension = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
        lm_extension = [".json"]

        if self.NumberScan(scan_folder) == 0 :
            return 'The selected folder must contain scans'
        
        dic = super().search(scan_folder,scan_extension,lm_extension)

        patients = self.PatientScanLandmark(dic,scan_extension,lm_extension)

        for patient,data in patients.items():
            if "scan" not in data.keys():
                out += "Missing scan for patient : {}\nat {}\n".format(patient,data["dir"])
            if len(data['lmrk']) == 0:
                out += "Missing landmark for patient : {}\nat {}\n".format(patient,data["dir"])
        
        if out == '':   # If no errors
            out = None
        return out

    def existsLandmark(self, input_dir, reference_dir, model_dir):
        out = None
        if input_dir != '' and reference_dir != '':
            input_lm = []
            input_json = super().search(input_dir,'json')['json']

            # all_lm = [self.ListLandmarksJson(file) for file in input_json]
            # input_lm = all_lm[0]
            # for lm_file in all_lm:
            #     for lm in input_lm:
            #         if lm not in lm_file:
            #             input_lm.remove(lm)

            gold_json = super().search(reference_dir,'json')['json']
            gold_lm = self.ListLandmarksJson(gold_json[0])
            
            available_lm = [lm for lm in gold_lm]#input_lm if lm in gold_lm]
            available = {key:True for key in available_lm}
            
            dic = self.DicLandmark()['Landmark']
            list_lm = []
            for key in dic.keys():
                list_lm.extend(dic[key])

            not_available_lm = [lm for lm in list_lm if lm not in available_lm]
            not_available = {key:False for key in not_available_lm} 
            
            out = {**available,**not_available}

        return out

    def Process(self, **kwargs):
        list_lmrk_str = self.CheckboxisChecked(kwargs['dic_checkbox'],in_str=True)
       
        parameter_semi_aso= {'input':kwargs['input_folder'],
                    'gold_folder':kwargs['gold_folder'],
                    'output_folder':kwargs['folder_output'],
                    'add_inname':kwargs['add_in_namefile'],
                    'list_landmark':list_lmrk_str
                }
        print('parameter',parameter_semi_aso)

        OrientProcess = slicer.modules.semi_aso_cbct
        list_process = [{'Process':OrientProcess,'Parameter':parameter_semi_aso}]

        nb_scan = self.NumberScan(kwargs['input_folder'])
        display =  {'SEMI_ASO_CBCT':DisplayASOCBCT(nb_scan)}
        
        return list_process, display

class Auto_CBCT(CBCT):

    def getTestFileList(self):
        return ("Fully-Automated", "https://github.com/lucanchling/ASO_CBCT/releases/download/TestFiles/Test_Scan.zip")
        
    def TestScan(self, scan_folder: str) -> str:
        return None

    def existsLandmark(self, input_dir, reference_dir, model_dir):
        out = None

        if reference_dir != '' and model_dir != '':

            gold_json = super().search(reference_dir,'json')['json']
            gold_lm = self.ListLandmarksJson(gold_json[0])

            list_model_files = super().search(model_dir,'pth')['pth']
            list_models = [os.path.basename(i).split('_Net')[0] for i in list_model_files]
            
            available_lm = [lm for lm in gold_lm if lm in list_models]
            available = {key:True for key in available_lm}
            
            dic = self.DicLandmark()['Landmark']
            list_lm = []
            for key in dic.keys():
                list_lm.extend(dic[key])

            not_available_lm = [lm for lm in list_lm if lm not in available_lm]
            not_available = {key:False for key in not_available_lm} 
            
            out = {**available,**not_available}

        return out

    def Process(self, **kwargs):

        # PRE ASO CBCT
        temp_folder = slicer.util.tempDirectory()
        time.sleep(0.01)
        tempPREASO_folder = slicer.util.tempDirectory()
        parameter_pre_aso = {'input': kwargs['input_folder'],
                             'output_folder': temp_folder,#kwargs['input_folder'],
                             'model_folder':kwargs['model_folder_segor'],
                             'SmallFOV':kwargs['smallFOV'],
                             'temp_folder': tempPREASO_folder}
        
        PreOrientProcess = slicer.modules.pre_aso_cbct

        list_lmrk_str = self.CheckboxisChecked(kwargs['dic_checkbox'],in_str=True)
        nb_landmark = len(list_lmrk_str.split(' '))

        print('PRE_ASO param:', parameter_pre_aso)
        print()

        # ALI CBCT
        documentsLocation = qt.QStandardPaths.DocumentsLocation
        documents = qt.QStandardPaths.writableLocation(documentsLocation)
        tempALI_folder = os.path.join(documents, slicer.app.applicationName+"_temp_ALI")
        
        parameter_ali =  {'input': temp_folder, 
                    'dir_models': kwargs['model_folder_ali'], 
                    'landmarks': list_lmrk_str, 
                    'save_in_folder': False, 
                    'output_dir': temp_folder,
                    'temp_fold': tempALI_folder,
                    'DCMInput':False}
        ALIProcess = slicer.modules.ali_cbct
        
        print('ALI param:',parameter_ali)
        print()
        # SEMI ASO CBCT        
       
        parameter_semi_aso = {'input':temp_folder,#kwargs['input_folder'],
                    'gold_folder':kwargs['gold_folder'],
                    'output_folder':kwargs['folder_output'],
                    'add_inname':kwargs['add_in_namefile'],
                    'list_landmark':list_lmrk_str,
                }
        OrientProcess = slicer.modules.semi_aso_cbct

        print("SEMI_ASO param:",parameter_semi_aso)
 
        list_process = [{'Process':PreOrientProcess,'Parameter':parameter_pre_aso,'Name':'PRE_ASO_CBCT'},
                        {'Process':ALIProcess,'Parameter': parameter_ali,'Name':'ALI_CBCT'},
                        {'Process':OrientProcess,'Parameter':parameter_semi_aso,'Name':'SEMI_ASO_CBCT'}
        ]
        nb_scan = self.NumberScan(kwargs['input_folder'])

        display = {'ALI_CBCT':DisplayALICBCT(nb_landmark,nb_scan),
                   'SEMI_ASO_CBCT':DisplayASOCBCT(nb_scan),
                   'PRE_ASO_CBCT':DisplayASOCBCT(nb_scan)}

        return list_process, display
        
        
    