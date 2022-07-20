# Slicer Automated Dental Tools 

_Authors: Maxime Gillot (University of Michigan), Baptiste Baquero (UoM)_


Slicer automated dental tools is an extension perform automatic **segmentation** and **landmark identification** using machine learning tools.
<img src="SlicerAutomaticTools.png" alt="Extension Logo" width="100"/>



This extension will allow you to :
- segment CBCT scan using [AMASSS](https://github.com/Maxlo24/AMASSS_CBCT)
- Identify landmarks in CBCT using [ALI-CBCT](https://github.com/Maxlo24/ALI_CBCT)
- Identify landmarks in IOS using [ALI-IOS](https://github.com/baptistebaquero/ALIDDM)


## Requirements 

```
Slicer 5.1.0 or later
12GB of VRAM if possible
```

### :warning: Warning

- All modules relies on machine learning tools that **requires GPU** for optimal use
- The user will have to **download the trained networks**  required for each module

---

<img src="ADT-exemple.png" alt="Exemples"/>
# How to use the modules

On slicer, in the module selection table, a new option named **"Automated dental tools"** will allo you to choose between the modules :
- AMASSS (Automatic Multi-Anatomical Skull Structure Segmentation)
- ALI (Automatic Landmark Identification)

Both modules shares common features :

**Input**
- All modules can work with one file or a whole sample (folder) as input.
- If the input is a single file already loaded on slicer, the result of the predicton will directly show up on the slicer views.

**Output**
- By selecting the "Group output in a folder" checkbox, all the ouput files will be grouped in a single folder for each patient.

- All modules allows the user to save the output in the input folder, or by unchecking the "Save prediction in scan folder" the user can choose a custom output folder.

- The "Prediction ID" field is for the user to choose what will appear on the output file name. ("Pred" by default) 

---

## AMASSS Module
<img src="AMASSS/Resources/Icons/AMASSS.png" alt="Extension Logo" width="50"/>

AMASSS module will allow you to segment CBCT scan using [AMASSS](https://github.com/Maxlo24/AMASSS_CBCT) algortihm.

**Input file:**
The input has to be an oriented CBCT.
It can be a single CBCT scan loaded on slicer or a folder containg CBCTs with the following extention:
```
.nrrd / .nrrd.gz
.nii  / .nii.gz
.gipl / .gipl.gz
```
**Load models:**
The user has to indicate the path of the folder containing the [trained models for AMASSS](https://github.com/Maxlo24/AMASSS_CBCT/releases/download/v1.0.1/ALL_NEW_MODELS.zip).


**Segmentation selection:**
The user can choose the structure to segment using the selection table.
Depending on the type of CBCT to segment, the user can select the "Use small FOV models" checkbox to use on higher definition scans.
![SegTab](https://user-images.githubusercontent.com/46842010/180010213-242a7e35-a16c-4ccd-b0c2-aaadae9b15c6.png)


**Output option:**
By selecting the **"Generate surface files"** checkbox. The user will also get a **surface model of the segmentation** that will be saved in a "VTK files" folder and will be automatically loaded in slicer at the end of the prediction if **working on a single file**.

**Advanced option:**
- You can increase/decrease the precision of the segmentation (going above 50 will drastically increase the prediction time and is not necesary worth it, going under 50 will make the prediction much faster but less accurate)
- If the user whant to generate surface files, he can choose the smothness applied on the model.
- Depending on your computer power, you can increase the CPU and GPU usage to increase the predictio speed.

---

## ALI Module
<img src="ALI/Resources/Icons/ALI.png" alt="Extension Logo" width="50"/>

ALI module as 2 modes that will allow the user to identify landmarks on:
- CBCT scan using [ALI-CBCT](https://github.com/Maxlo24/ALI_CBCT) algortihm.
- IOS scan using [ALI-IOS](https://github.com/baptistebaquero/ALIDDM) algortihm.



### ALI-CBCT
**Input file:**
The input has to be an oriented CBCT.
It can be a single CBCT scan loaded on slicer or a folder containg CBCTs with the following extention:
```
.nrrd / .nrrd.gz
.nii  / .nii.gz
.gipl / .gipl.gz
```
**Load models:**
The user has to indicate the path of the folder containing the [trained models for ALI-CBCT](https://github.com/Maxlo24/ALI_CBCT/releases/tag/v0.1-models).

**Landmark selection:**
Once the folder containing the trained models is loaded. The user can choose the landmark he want to identify with the table showing the available landmarks:
![SegTab](https://user-images.githubusercontent.com/46842010/180010603-37dce4c3-e7f8-4b3a-98a1-2874918320cb.png)

---

### ALI-IOS

**Input file:**
The input has to be an oriented IOS segmented with the [Universal Numbering System](https://en.wikipedia.org/wiki/Universal_Numbering_System).
This segmentation can be automatically done using the [SlicerJawSegmentation](https://github.com/MathieuLeclercq/SlicerJawSegmentation) extention.
The input can be a single IOS loaded on slicer or a folder containg IOS with the following extention:
```
.vtk
```

**Load models:**
The user has to indicate the path of the folder containing the [trained models for ALI-IOS](https://github.com/baptistebaquero/ALIDDM/releases/tag/v1.0.3).

**Landmark selection:**
For the IOS landmarks, the user has to choose which tooth he need the landmakrs on by checking the label of the tooth on the left table.
Once the folder containing the trained models is loaded. The user can choose the landmark he want to identify with the table on the right showing the available landmarks:
![LM_tab_ios](https://user-images.githubusercontent.com/46842010/180010083-4f7b6e31-edd3-41a2-a696-6a6a1a4d9260.png)


