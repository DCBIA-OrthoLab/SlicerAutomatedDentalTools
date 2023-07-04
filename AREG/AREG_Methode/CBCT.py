from AREG_Methode.Methode import Methode
from AREG_Methode.Progress import DisplayAREGCBCT, DisplayAMASSS, DisplayALICBCT, DisplayASOCBCT
import os,sys

from glob import iglob
import slicer
import time
import qt
import platform

class Semi_CBCT(Methode):
    def __init__(self, widget):
        super().__init__(widget)
        documentsLocation = qt.QStandardPaths.DocumentsLocation
        documents = qt.QStandardPaths.writableLocation(documentsLocation)
        self.tempAMASSS_folder = os.path.join(documents, slicer.app.applicationName+"_temp_AMASSS")
        self.tempALI_folder = os.path.join(documents, slicer.app.applicationName+"_temp_ALI")

    def getGPUUsage(self):
        if platform.system() == 'Darwin':
            return 1
        else:
            return 5

    def NumberScan(self, scan_folder_t1: str,scan_folder_t2: str):
        return len(GetDictPatients(scan_folder_t1, scan_folder_t2))
        
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
        
        return None

    def TestCheckbox(self,dic_checkbox):
        list_landmark = self.CheckboxisChecked(dic_checkbox)['Regions of Reference for Registration']
        
        out = None
        if len(list_landmark) == 0:
             out = 'Please select a Registration Type\n'
        return out

    def TestModel(self, model_folder: str,lineEditName) -> str:

        if lineEditName == 'lineEditModel1':
            if len(super().search(model_folder,'pth')['pth']) == 0:
                return 'Folder must have AMASSS masks models files'
            else:
                return None
        
        # if lineEditName == 'lineEditModelAli':
        #     if len(super().search(model_folder,'pth')['pth']) == 0:
        #         return 'Folder must have ALI models files'
        #     else:
        #         return None

    def TestProcess(self, **kwargs) -> str:
        out=''

        testcheckbox = self.TestCheckbox(kwargs['dic_checkbox'])
        if testcheckbox is not None:
            out+=testcheckbox

        if kwargs['input_t1_folder'] == '':
            out+= 'Please select an input folder for T1 scans\n'

        if kwargs['input_t2_folder'] == '':
            out+= 'Please select an input folder for T2 scans\n'

        if kwargs['folder_output'] == '':
            out+= 'Please select an output folder\n'

        if kwargs['add_in_namefile']== '':
            out += 'Please select an extension for output files\n'

        if kwargs['model_folder_1'] == '':
            out += 'Please select a folder for AMASSS models\n'

        if out == '':
            out = None

        return out

    def getModelUrl(self):
        return {"AMASSS":{"Full Face Models":"https://github.com/lucanchling/AMASSS_CBCT/releases/download/v1.0.2/AMASSS_Models.zip","Mask Models":"https://github.com/lucanchling/AMASSS_CBCT/releases/download/v1.0.2/Masks_Models.zip"}}

    def getALIModelList(self):
        return ("ALIModels", "https://github.com/lucanchling/ALI_CBCT/releases/download/models_v01/")


    def ProperPrint(self,notfound_list):
        dic = {'scanT1':'T1 scan','scanT2':'T2 scan','segT1':'T1 segmentation','segT2':'T2 segmentation'}
        out = ''
        if 'scanT1' in notfound_list and 'scanT2' in notfound_list:
            out += 'T1 and T2 scans\n'
        elif 'segT1' in notfound_list and 'segT2' in notfound_list:
            out += 'T1 and T2 segmentations\n'
        else:
            for notfound in notfound_list:
                out += dic[notfound]+' '
        return out
    
    def TestScan(self, scan_folder_t1: str, scan_folder_t2: str, liste_keys = ['scanT1','scanT2','segT1']):
        out = ''
        scan_extension = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
        if self.NumberScan(scan_folder_t1,scan_folder_t2) == 0 :
            return 'Please Select folder with scans'
        
        patients = GetDictPatients(scan_folder_t1,scan_folder_t2)
        for patient,data in patients.items():
            not_found = [key for key in liste_keys if key not in data.keys()]
            if len(not_found) != 0:
                out += f"Patient {patient} does not have {self.ProperPrint(not_found)}\n"

        if out == '':   # If no errors
            out = None
            
        return out

        

    def CheckboxisChecked(self,diccheckbox : dict, in_str = False):
        listchecked = {key:[] for key in diccheckbox.keys()}
        for key,checkboxs in diccheckbox.items():
            for checkbox in checkboxs:
                if checkbox.isChecked():
                    listchecked[key] += [checkbox.text]

        # if not len(diccheckbox) == 0:
        #     for checkboxs in diccheckbox.values():
        #         for checkbox in checkboxs:
        #             if checkbox.isChecked():
        #                 listchecked.append(checkbox.text)
        # if in_str:
        #     listchecked_str = ''
        #     for i,lm in enumerate(listchecked):
        #         if i<len(listchecked)-1:
        #             listchecked_str+= lm+' '
        #         else:
        #             listchecked_str+=lm
        #     return listchecked_str
    
        return listchecked
    
    def DicLandmark(self):
        return {'Regions of Reference for Registration':["Cranial Base","Mandible","Maxilla"],
                'AMASSS Segmentation':['Cranial Base','Cervical Vertebra','Mandible','Maxilla','Skin','Upper Airway']}

    def TranslateModels(self,listeModels,mask=False):
        dicTranslate = {
            "Models": {
                "Mandible" : "MAND",
                "Maxilla" : "MAX",
                "Cranial Base" : "CB",
                "Cervical Vertebra" : "CV",
                "Root Canal" : "RC",
                "Mandibular Canal" : "MCAN",
                "Upper Airway" : "UAW",
                "Skin" : "SKIN",
            },
            "Masks": {
                "Cranial Base": "CBMASK",
                "Mandible": "MANDMASK",
                "Maxilla": "MAXMASK",
            }
        }
        
        translate = ""
        for i,model in enumerate(listeModels):
            if i<len(listeModels)-1:
                if mask:
                    translate += dicTranslate["Masks"][model]+" "
                else:
                    translate += dicTranslate["Models"][model]+" "
            else:
                if mask:
                    translate += dicTranslate["Masks"][model]
                else:
                    translate += dicTranslate["Models"][model]

        return translate
    
    def existsLandmark(self, input_dir, reference_dir, model_dir):
        return None
    

    def getTestFileList(self):
        return ("Semi-Automated", "https://github.com/lucanchling/Areg_CBCT/releases/download/TestFiles/SemiAuto.zip")


    def Process(self, **kwargs):
        list_struct = self.CheckboxisChecked(kwargs['dic_checkbox'])
        full_reg_struct,full_seg_struct = list_struct['Regions of Reference for Registration'],list_struct['AMASSS Segmentation']
        reg_struct,seg_struct = self.TranslateModels(full_reg_struct, False),self.TranslateModels(full_seg_struct, False)

        nb_scan = self.NumberScan(kwargs['input_t1_folder'],kwargs['input_t2_folder'])

        centered_T2 = kwargs['input_t2_folder'] + '_Center'
        parameter_pre_aso = {'input': kwargs['input_t2_folder'],
                             'output_folder': centered_T2,
                             'model_folder':os.path.join(kwargs['slicerDownload'],'Models','Orientation','PreASO'),
                             'SmallFOV':False,
                             'temp_folder': '../',
                             'DCMInput':kwargs['isDCMInput'],
                             }
        
        PreOrientProcess = slicer.modules.pre_aso_cbct
        list_process = [{'Process':PreOrientProcess,'Parameter':parameter_pre_aso,'Module':'Centering T2','Display': DisplayASOCBCT(nb_scan)}]

        # AREG CBCT PROCESS
        AREGProcess = slicer.modules.areg_cbct
        for i,reg in enumerate(reg_struct.split(' ')):
            parameter_areg_cbct = {
                        't1_folder':kwargs['input_t1_folder'],
                        't2_folder':centered_T2,
                        'reg_type':reg,
                        'output_folder':kwargs['folder_output'],
                        'add_name':kwargs['add_in_namefile'],
                        'DCMInput':False,
                    }
            list_process.append({'Process':AREGProcess,'Parameter':parameter_areg_cbct,'Module':'AREG_CBCT for {}'.format(full_reg_struct[i]),'Display':DisplayAREGCBCT(nb_scan)})
        
        # AMASSS PROCESS - SEGMENTATION
        AMASSSProcess = slicer.modules.amasss_cli
        parameter_amasss_seg_t1 = {"inputVolume": kwargs['input_t1_folder'],
                                "modelDirectory": kwargs['model_folder_1'],
                                "highDefinition": False,
                                "skullStructure": seg_struct,
                                "merge": "MERGE" if kwargs['merge_seg'] else "SEPARATE",
                                "genVtk": True,
                                "save_in_folder": True,
                                "output_folder": kwargs['folder_output'],
                                "precision": 50,
                                "vtk_smooth": 5,
                                "prediction_ID": 'Pred',
                                "gpu_usage": self.getGPUUsage(),
                                "cpu_usage": 1,
                                "temp_fold" : self.tempAMASSS_folder,
                                "SegmentInput" : False,
                                "DCMInput": False,
        }              
        parameter_amasss_seg_t2 = {"inputVolume": kwargs['folder_output'],
                                "modelDirectory": kwargs['model_folder_1'],
                                "highDefinition": False,
                                "skullStructure": seg_struct,
                                "merge": "MERGE" if kwargs['merge_seg'] else "SEPARATE",
                                "genVtk": True,
                                "save_in_folder": True,
                                "output_folder": kwargs['folder_output'],
                                "precision": 50,
                                "vtk_smooth": 5,
                                "prediction_ID": 'Pred',
                                "gpu_usage": self.getGPUUsage(),
                                "cpu_usage": 1,
                                "temp_fold" : self.tempAMASSS_folder,
                                "SegmentInput" : False,
                                "DCMInput": False,
        }
        if len(full_seg_struct) > 0:
            list_process.append({'Process':AMASSSProcess,'Parameter':parameter_amasss_seg_t1,'Module':'AMASSS_CBCT Segmentation of T1','Display':DisplayAMASSS(nb_scan,len(full_seg_struct))})
            list_process.append({'Process':AMASSSProcess,'Parameter':parameter_amasss_seg_t2,'Module':'AMASSS_CBCT Segmentation of T2','Display':DisplayAMASSS(nb_scan,len(full_seg_struct),len(full_reg_struct))})
        
        return list_process


class Auto_CBCT(Semi_CBCT):

    def getTestFileList(self):
        return ("Fully-Automated", "https://github.com/lucanchling/Areg_CBCT/releases/download/TestFiles/FullyAuto.zip")
    
    def TestScan(self, scan_folder_t1: str, scan_folder_t2: str):
        return super().TestScan(scan_folder_t1, scan_folder_t2, liste_keys = ['scanT1','scanT2'])

    def Process(self, **kwargs):

        list_struct = self.CheckboxisChecked(kwargs['dic_checkbox'])
        
        full_reg_struct = list_struct['Regions of Reference for Registration']
        reg_struct = self.TranslateModels(full_reg_struct, True)
        
        nb_scan = self.NumberScan(kwargs['input_t1_folder'], kwargs['input_t2_folder'])
        
        # AMASSS PROCESS - MASK SEGMENTATIONS        
        parameter_amasss_mask_t1 = {"inputVolume": kwargs['input_t1_folder'],
                                "modelDirectory": kwargs['model_folder_1'],
                                "highDefinition": False,
                                "skullStructure": reg_struct,
                                "merge": "SEPARATE",
                                "genVtk": False,
                                "save_in_folder": False,
                                "output_folder": kwargs['input_t1_folder'],
                                "precision": 50,
                                "vtk_smooth": 5,
                                "prediction_ID": 'Pred',
                                "gpu_usage": self.getGPUUsage(),
                                "cpu_usage": 1,
                                "temp_fold" : self.tempAMASSS_folder,
                                "SegmentInput" : False,
                                "DCMInput": False,
        }
        AMASSSProcess = slicer.modules.amasss_cli
        list_process = [{'Process':AMASSSProcess,'Parameter':parameter_amasss_mask_t1, 'Module':'AMASSS_CBCT - Masks Generation for T1', 'Display': DisplayAMASSS(nb_scan, len(full_reg_struct))},
        ]

        # print('AMASSS Mask Parameters:', parameter_amasss_mask_t1)
        # print()
        centered_T2 = kwargs['input_t2_folder'] + '_Center'
        parameter_pre_aso = {'input': kwargs['input_t2_folder'],
                             'output_folder': centered_T2,
                             'model_folder':os.path.join(kwargs['slicerDownload'],'Models','Orientation','PreASO'),
                             'SmallFOV':False,
                             'temp_folder': '../',
                             'DCMInput':kwargs['isDCMInput'],
                             }
        
        PreOrientProcess = slicer.modules.pre_aso_cbct
        list_process.append({'Process':PreOrientProcess,'Parameter':parameter_pre_aso,'Module':'Centering T2','Display': DisplayASOCBCT(nb_scan)})

        # AREG CBCT PROCESS
        full_reg_struct = list_struct['Regions of Reference for Registration']
        reg_struct = self.TranslateModels(full_reg_struct, False)
        
        AREGProcess = slicer.modules.areg_cbct
        for i,reg in enumerate(reg_struct.split(' ')):
            parameter_areg_cbct = {
                        't1_folder':kwargs['input_t1_folder'],
                        't2_folder':centered_T2,
                        'reg_type':reg,
                        'output_folder':kwargs['folder_output'],
                        'add_name':kwargs['add_in_namefile'],
                        'DCMInput':False,
                    }
            list_process.append({'Process':AREGProcess,'Parameter':parameter_areg_cbct,'Module':'AREG_CBCT for {}'.format(full_reg_struct[i]),'Display': DisplayAREGCBCT(nb_scan)})

        full_seg_struct = list_struct['AMASSS Segmentation']
        seg_struct = self.TranslateModels(full_seg_struct, False)

        # AMASSS PROCESS - SEGMENTATIONS
        AMASSSProcess = slicer.modules.amasss_cli
        parameter_amasss_seg_t1 = {"inputVolume": kwargs['input_t1_folder'],
                                "modelDirectory": kwargs['model_folder_1'],
                                "highDefinition": False,
                                "skullStructure": seg_struct,
                                "merge": "MERGE" if kwargs['merge_seg'] else "SEPARATE",
                                "genVtk": True,
                                "save_in_folder": False,
                                "output_folder": kwargs['folder_output'],
                                "precision": 50,
                                "vtk_smooth": 5,
                                "prediction_ID": 'Pred',
                                "gpu_usage": self.getGPUUsage(),
                                "cpu_usage": 1,
                                "temp_fold" : self.tempAMASSS_folder,
                                "SegmentInput" : False,
                                "DCMInput": False,
        }        
        parameter_amasss_seg_t2 = {"inputVolume": kwargs['folder_output'],
                                "modelDirectory": kwargs['model_folder_1'],
                                "highDefinition": False,
                                "skullStructure": seg_struct,
                                "merge": "MERGE" if kwargs['merge_seg'] else "SEPARATE",
                                "genVtk": True,
                                "save_in_folder": False,
                                "output_folder": kwargs['folder_output'],
                                "precision": 50,
                                "vtk_smooth": 5,
                                "prediction_ID": 'Pred',
                                "gpu_usage": self.getGPUUsage(),
                                "cpu_usage": 1,
                                "temp_fold" : self.tempAMASSS_folder,
                                "SegmentInput" : False,
                                "DCMInput": False,
        }
        if len(full_seg_struct) > 0:
            list_process.append({'Process':AMASSSProcess,'Parameter':parameter_amasss_seg_t1,'Module':'AMASSS_CBCT Segmentation for T1','Display':DisplayAMASSS(nb_scan,len(full_seg_struct))})
            list_process.append({'Process':AMASSSProcess,'Parameter':parameter_amasss_seg_t2,'Module':'AMASSS_CBCT Segmentation for T2','Display':DisplayAMASSS(nb_scan,len(full_seg_struct),len(full_reg_struct))})

        return list_process

class Or_Auto_CBCT(Semi_CBCT):

    def getModelUrl(self):
        return {
            "AMASSS": 
                {"Full Face Models":"https://github.com/lucanchling/AMASSS_CBCT/releases/download/v1.0.2/AMASSS_Models.zip",
                 "Mask Models":"https://github.com/lucanchling/AMASSS_CBCT/releases/download/v1.0.2/Masks_Models.zip"},
            "Orientation" : 
                {"PreASO":"https://github.com/lucanchling/ASO_CBCT/releases/download/v01_preASOmodels/PreASOModels.zip",
                 "Reference":"https://github.com/lucanchling/ASO_CBCT/releases/download/v01_goldmodels/Frankfurt_Horizontal_Midsagittal_Plane.zip"
                 }
                 }

    def getTestFileList(self):
        return ("Oriented-Automated", "https://github.com/lucanchling/Areg_CBCT/releases/download/TestFiles/Or_FullyAuto.zip")
    
    def getTestFileListDCM(self):
        return ("Oriented-Automated", "https://github.com/lucanchling/Areg_CBCT/releases/download/TestFiles/Or_FullyAuto_DCM.zip")
    
    def TestScan(self, scan_folder_t1: str, scan_folder_t2: str):
        return super().TestScan(scan_folder_t1, scan_folder_t2, liste_keys = ['scanT1','scanT2'])

    def TestScanDCM(self, scan_folder_t1: str, scan_folder_t2) -> str:
        out = ''
        liste_t1 = [folder for folder in os.listdir(scan_folder_t1) if os.path.isdir(os.path.join(scan_folder_t1,folder)) and folder != 'NIFTI']
        liste_t2 = [folder for folder in os.listdir(scan_folder_t2) if os.path.isdir(os.path.join(scan_folder_t2,folder)) and folder != 'NIFTI']
        some = False
        for pat in liste_t2:
            if pat not in liste_t1:        
                out+= 'T1 folder --> patient {} is missing\n'.format(pat)
                some = True

        if some:
            out+='---------------------------------------------------------------\n'
        for pat in liste_t1:
            if pat not in liste_t2:
                out+= 'T2 folder --> patient {} is missing\n'.format(pat)
        
        if out == '':
            out = None
        
        return out

    def NumberScanDCM(self, scan_folder_t1: str, scan_folder_t2: str):
        return len(os.listdir(scan_folder_t1))

    def TestProcess(self, **kwargs) -> str:
        out=''

        testcheckbox = self.TestCheckbox(kwargs['dic_checkbox'])
        if testcheckbox is not None:
            out+=testcheckbox

        if kwargs['input_t1_folder'] == '':
            out+= 'Please select an input folder for T1 scans\n'

        if kwargs['input_t2_folder'] == '':
            out+= 'Please select an input folder for T2 scans\n'

        if kwargs['folder_output'] == '':
            out+= 'Please select an output folder\n'

        if kwargs['add_in_namefile']== '':
            out += 'Please select an extension for output files\n'

        if kwargs['model_folder_1'] == '':
            out += 'Please download the AMASSS models\n'

        if kwargs['model_folder_2'] == '':
            out += 'Please download the Orientation folder\n'

        if out == '':
            out = None

        return out
    
    def Process(self, **kwargs):

        # ====================== ASO Process ======================
        # PRE ASO CBCT
        temp_folder = slicer.util.tempDirectory()
        time.sleep(0.01)
        tempPREASO_folder = slicer.util.tempDirectory()
        parameter_pre_aso = {'input': kwargs['input_t1_folder'],
                             'output_folder': temp_folder,#kwargs['input_folder'],
                             'model_folder':os.path.join(kwargs['model_folder_2'],'PreASO'),
                             'SmallFOV':False,
                             'temp_folder': tempPREASO_folder,
                             'DCMInput':kwargs['isDCMInput'],
                             }
        
        PreOrientProcess = slicer.modules.pre_aso_cbct

        list_lmrk_str = "N S Ba RPo LPo LOr ROr"
        nb_landmark = 7

        print('PRE_ASO param:', parameter_pre_aso)
        print()

        # ALI CBCT
        parameter_ali =  {'input': temp_folder, 
                    'dir_models': os.path.join(kwargs['model_folder_2'],"ALIModels"), 
                    'landmarks': list_lmrk_str, 
                    'save_in_folder': False, 
                    'output_dir': temp_folder,
                    'temp_fold': self.tempALI_folder,
                    'DCMInput':False}
        ALIProcess = slicer.modules.ali_cbct
        
        print('ALI param:',parameter_ali)
        print()
        # SEMI ASO CBCT
        ASO_T1_Oriented = kwargs['input_t1_folder']+'Or'
        parameter_semi_aso = {'input':temp_folder,#kwargs['input_folder'],
                    'gold_folder':os.path.join(kwargs['model_folder_2'],'Reference'),
                    'output_folder':ASO_T1_Oriented,
                    'add_inname':'Or',
                    'list_landmark':list_lmrk_str,
                }
        OrientProcess = slicer.modules.semi_aso_cbct

        print("SEMI_ASO param:",parameter_semi_aso)
 
        nb_scan = self.NumberScan(kwargs['input_t1_folder'], kwargs['input_t2_folder']) if not kwargs['isDCMInput'] else self.NumberScanDCM(kwargs['input_t1_folder'], kwargs['input_t2_folder'])
        list_process = [{'Process':PreOrientProcess,'Parameter':parameter_pre_aso,'Module':'PRE_ASO_CBCT','Display':DisplayASOCBCT(nb_scan)},
                        {'Process':ALIProcess,'Parameter': parameter_ali,'Module':'ALI_CBCT','Display':DisplayALICBCT(nb_landmark,nb_scan)},
                        {'Process':OrientProcess,'Parameter':parameter_semi_aso,'Module':'SEMI_ASO_CBCT','Display':DisplayASOCBCT(nb_scan)}
        ]

        # ====================== AREG Process ======================
        list_struct = self.CheckboxisChecked(kwargs['dic_checkbox'])
        
        full_reg_struct = list_struct['Regions of Reference for Registration']
        reg_struct = self.TranslateModels(full_reg_struct, True)
        
        # AMASSS PROCESS - MASK SEGMENTATIONS        
        parameter_amasss_mask_t1 = {"inputVolume": ASO_T1_Oriented,
                                "modelDirectory": kwargs['model_folder_1'],
                                "highDefinition": False,
                                "skullStructure": reg_struct,
                                "merge": "SEPARATE",
                                "genVtk": False,
                                "save_in_folder": False,
                                "output_folder": kwargs['input_t1_folder'],
                                "precision": 50,
                                "vtk_smooth": 5,
                                "prediction_ID": 'Pred',
                                "gpu_usage": self.getGPUUsage(),
                                "cpu_usage": 1,
                                "temp_fold" : self.tempAMASSS_folder,
                                "SegmentInput" : False,
                                "DCMInput": False,
        }
        AMASSSProcess = slicer.modules.amasss_cli
        list_process += [{'Process':AMASSSProcess,'Parameter':parameter_amasss_mask_t1, 'Module':'AMASSS_CBCT - Masks Generation for T1', 'Display': DisplayAMASSS(nb_scan, len(full_reg_struct))}]

        # print('AMASSS Mask Parameters:', parameter_amasss_mask_t1)
        # print()
        centered_T2 = kwargs['input_t2_folder'] + '_Center'
        parameter_pre_aso = {'input': kwargs['input_t2_folder'],
                             'output_folder': centered_T2,
                             'model_folder':os.path.join(kwargs['model_folder_2'],'PreASO'),
                             'SmallFOV':False,
                             'temp_folder': '../',
                             'DCMInput':kwargs['isDCMInput'],
                             }
        
        PreOrientProcess = slicer.modules.pre_aso_cbct
        list_process.append({'Process':PreOrientProcess,'Parameter':parameter_pre_aso,'Module':'Centering T2','Display': DisplayASOCBCT(nb_scan)})


        # AREG CBCT PROCESS
        full_reg_struct = list_struct['Regions of Reference for Registration']
        reg_struct = self.TranslateModels(full_reg_struct, False)
        
        AREGProcess = slicer.modules.areg_cbct
        for i,reg in enumerate(reg_struct.split(' ')):
            parameter_areg_cbct = {
                        't1_folder':ASO_T1_Oriented,
                        't2_folder':centered_T2,
                        'reg_type':reg,
                        'output_folder':kwargs['folder_output'],
                        'add_name':kwargs['add_in_namefile'],
                        'DCMInput':kwargs['isDCMInput'],
                    }
            list_process.append({'Process':AREGProcess,'Parameter':parameter_areg_cbct,'Module':'AREG_CBCT for {}'.format(full_reg_struct[i]),'Display': DisplayAREGCBCT(nb_scan)})

        full_seg_struct = list_struct['AMASSS Segmentation']
        seg_struct = self.TranslateModels(full_seg_struct, False)

        # AMASSS PROCESS - SEGMENTATIONS
        AMASSSProcess = slicer.modules.amasss_cli
        parameter_amasss_seg_t1 = {"inputVolume": ASO_T1_Oriented,
                                "modelDirectory": kwargs['model_folder_1'],
                                "highDefinition": False,
                                "skullStructure": seg_struct,
                                "merge": "MERGE" if kwargs['merge_seg'] else "SEPARATE",
                                "genVtk": True,
                                "save_in_folder": False,
                                "output_folder": kwargs['folder_output'],
                                "precision": 50,
                                "vtk_smooth": 5,
                                "prediction_ID": 'Pred',
                                "gpu_usage": self.getGPUUsage(),
                                "cpu_usage": 1,
                                "temp_fold" : self.tempAMASSS_folder,
                                "SegmentInput" : False,
                                "DCMInput": False,
        }        
        parameter_amasss_seg_t2 = {"inputVolume": kwargs['folder_output'],
                                "modelDirectory": kwargs['model_folder_1'],
                                "highDefinition": False,
                                "skullStructure": seg_struct,
                                "merge": "MERGE" if kwargs['merge_seg'] else "SEPARATE",
                                "genVtk": True,
                                "save_in_folder": False,
                                "output_folder": kwargs['folder_output'],
                                "precision": 50,
                                "vtk_smooth": 5,
                                "prediction_ID": 'Pred',
                                "gpu_usage": self.getGPUUsage(),
                                "cpu_usage": 1,
                                "temp_fold" : self.tempAMASSS_folder,
                                "SegmentInput" : False,
                                "DCMInput": False,
        }
        if len(full_seg_struct) > 0:
            list_process.append({'Process':AMASSSProcess,'Parameter':parameter_amasss_seg_t1,'Module':'AMASSS_CBCT Segmentation for T1','Display':DisplayAMASSS(nb_scan,len(full_seg_struct))})
            list_process.append({'Process':AMASSSProcess,'Parameter':parameter_amasss_seg_t2,'Module':'AMASSS_CBCT Segmentation for T2','Display':DisplayAMASSS(nb_scan,len(full_seg_struct),len(full_reg_struct))})

        return list_process
    






def GetListNamesSegType(segmentationType):
    dic = {'CB':['cb'],
           'MAND':['mand','md'],
           'MAX':['max','mx'],}
    return dic[segmentationType]

def GetListFiles(folder_path, file_extension):
    """Return a list of files in folder_path finishing by file_extension"""
    file_list = []
    for extension_type in file_extension:
        file_list += search(folder_path,file_extension)[extension_type]
    return file_list

def GetPatients(folder_path, time_point='T1', segmentationType=None):
    """Return a dictionary with patient id as key"""
    file_extension = ['.nii.gz','.nii','.nrrd','.nrrd.gz','.gipl','.gipl.gz']
    json_extension = ['.json']
    file_list = GetListFiles(folder_path, file_extension+json_extension)

    patients = {}
    
    for file in file_list:
        basename = os.path.basename(file)
        patient = basename.split('_Scan')[0].split('_scan')[0].split('_Or')[0].split('_OR')[0].split('_MAND')[0].split('_MD')[0].split('_MAX')[0].split('_MX')[0].split('_CB')[0].split('_lm')[0].split('_T2')[0].split('_T1')[0].split('_Cl')[0].split('.')[0]
        
        if patient not in patients:
            patients[patient] = {}
        
        if True in [i in basename for i in file_extension]:
            # if segmentationType+'MASK' in basename:
            if True in [i in basename.lower() for i in ['mask','seg','pred']]:
                if segmentationType is None:
                    patients[patient]['seg'+time_point] = file
                else:
                    if True in [i in basename.lower() for i in GetListNamesSegType(segmentationType)]:
                        patients[patient]['seg'+time_point] = file
                
            else:
                patients[patient]['scan'+time_point] = file

        if True in [i in basename for i in json_extension]:
            if time_point == 'T2':
                patients[patient]['lm'+time_point] = file

    return patients

def GetMatrixPatients(folder_path):
    """Return a dictionary with patient id as key and matrix path as data"""
    file_extension = ['.tfm']
    file_list = GetListFiles(folder_path, file_extension)

    patients = {}
    for file in file_list:
        basename = os.path.basename(file)
        patient = basename.split("reg_")[1].split("_Cl")[0]
        if patient not in patients and True in [i in basename for i in file_extension]:
            patients[patient] = {}
            patients[patient]['mat'] = file  

    return patients

def GetDictPatients(folder_t1_path, folder_t2_path, segmentationType=None, todo_str='', matrix_folder=None):
    """Return a dictionary with patients for both time points"""
    patients_t1 = GetPatients(folder_t1_path, time_point='T1', segmentationType=segmentationType)
    patients_t2 = GetPatients(folder_t2_path, time_point='T2', segmentationType=None)
    patients = MergeDicts(patients_t1,patients_t2)
        
    if matrix_folder is not None:
        patient_matrix = GetMatrixPatients(matrix_folder)
        patients = MergeDicts(patients,patient_matrix)
    patients = ModifiedDictPatients(patients, todo_str)
    return patients

def MergeDicts(dict1,dict2):
    """Merge t1 and t2 dictionaries for each patient"""
    patients = {}
    for patient in dict1:
        patients[patient] = dict1[patient]
        try:
            patients[patient].update(dict2[patient])
        except KeyError:
            continue
    return patients


def ModifiedDictPatients(patients,todo_str):
    """Modify the dictionary of patients to only keep the ones in the todo_str"""

    if todo_str != '':
        liste_todo = todo_str.split(',')
        todo_patients = {}
        for i in liste_todo:
            patient = list(patients.keys())[int(i)-1]
            todo_patients[patient] = patients[patient]
        patients = todo_patients
    
    return patients



def search(path,*args):
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
    return {key: sorted([i for i in iglob(os.path.normpath("/".join([path,'**','*'])),recursive=True) if i.endswith(key)]) for key in arguments}