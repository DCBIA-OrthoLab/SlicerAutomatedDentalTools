# Slicer Automated Dental Tools 


Slicer automated dental tools is an extension for 3D Slicer to perform important automatic Dental and Cranio Facial analysis tasks via a GUI interface with no coding knowledge needed.


<p align="center">
    <img src="SlicerAutomaticTools.png" alt="Extension Logo" width="200"/>
</p>

## Overview

Slicer automated dental tools is an extension perform automatic **segmentation** and **landmark identification** on CBCT scans and Intra Oral Scan (IOS) using machine learning tools.


<p align="center">
<img src="ADT-exemple.png" alt="Exemples"/>
</p>

## Features


* **Simple**: Perform time consuming task with a few clicks.
* **Automatic**: Automate important Dental and Cranio Facial analysis tasks.
* **Flexible**: The user can choose which models to use to perform the automated task. New models can be added easily.


## Modules


| Name | Description |
|------|-------------|
| [AMASSS](AMASSS) | Perform automatic segmentation of CBCT scan using the [AMASSS](https://github.com/Maxlo24/AMASSS_CBCT) algorithm. |
| [ALI-CBCT](ALI_CBCT) | Perform automatic landmark identification in CBCT using [ALI-CBCT](https://github.com/Maxlo24/ALI_CBCT) algorithm. |
| [ALI-IOS](ALI_IOS) | Perform automatic landmark identification in IOS using [ALI-IOS](https://github.com/baptistebaquero/ALIDDM) algorithm. |





## Requirements 


* In addition of the [Slicer System requirements](https://slicer.readthedocs.io/en/latest/user_guide/getting_started.html#system-requirements), for best performance, 12GB of memory is recommended.
* :warning: Trained networks are required to be manually downloaded. See requirements section specific to each module.


<!-- ### :warning: IMPORTANT

- All modules relies on machine learning tools that **requires GPU** for optimal use
- The user will have to **download the trained networks**  required for each module
- _This extension is under active development. Its content, API and behavior may change at any time !_ -->
---


# How to use the modules



On slicer, in the module selection table, a new module category named **"Automated dental tools"** will allo you to choose between the modules :
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

---------------

## AMASSS Module
<img src="AMASSS/Resources/Icons/AMASSS.png" alt="Extension Logo" width="50"/>

AMASSS module will allow you to segment CBCT scan using [AMASSS](https://github.com/Maxlo24/AMASSS_CBCT) algortihm.


### Prerequisites

* Download the [trained models for AMASSS](https://github.com/Maxlo24/AMASSS_CBCT/releases/download/v1.0.1/ALL_NEW_MODELS.zip) using the `Download latest models` button in the module `Input section`.


### Module structure

**Input file:**
The input has to be an oriented CBCT.
It can be a single CBCT scan loaded on slicer or a folder containg CBCTs with the following extention:
```
.nrrd / .nrrd.gz
.nii  / .nii.gz
.gipl / .gipl.gz
```

Available sample data for testing: [MG_test_scan.nii.gz](https://github.com/Maxlo24/AMASSS_CBCT/releases/download/v1.0.1/MG_test_scan.nii.gz)

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

### Prerequisites

* Download the [trained models for ALI-CBCT](https://github.com/Maxlo24/ALI_CBCT/releases/tag/v0.1-models) using the `Download latest models` button in the module `Input section`.


### Module structure


**Input file:**
The input has to be an oriented CBCT.
It can be a single CBCT scan loaded on slicer or a folder containg CBCTs with the following extention:
```
.nrrd / .nrrd.gz
.nii  / .nii.gz
.gipl / .gipl.gz
```
Available sample data for testing: [MG_test_scan.nii.gz](https://github.com/Maxlo24/AMASSS_CBCT/releases/download/v1.0.1/MG_test_scan.nii.gz)

**Load models:**
The user has to indicate the path of the folder containing the [trained models for ALI-CBCT](https://github.com/Maxlo24/ALI_CBCT/releases/tag/v0.1-models).


**Landmark selection:**
Once the folder containing the trained models is loaded. The user can choose the landmark he want to identify with the table showing the available landmarks:
![SegTab](https://user-images.githubusercontent.com/46842010/180010603-37dce4c3-e7f8-4b3a-98a1-2874918320cb.png)

---

### ALI-IOS

### Prerequisites

* Download the [trained models for ALI-IOS](https://github.com/baptistebaquero/ALIDDM/releases/tag/v1.0.3) using the `Download latest models` button in the module `Input section`.


### Module structure


**Input file:**
The input has to be an oriented IOS segmented with the [Universal Numbering System](https://en.wikipedia.org/wiki/Universal_Numbering_System).
This segmentation can be automatically done using the [SlicerJawSegmentation](https://github.com/MathieuLeclercq/SlicerJawSegmentation) extention.
The input can be a single IOS loaded on slicer or a folder containg IOS with the following extention:
```
.vtk
```

Available sample data for testing: [T1_01_L_segmented.vtk](https://github.com/baptistebaquero/ALIDDM/releases/tag/v1.0.4) and [T1_01_U_segmented.vtk](https://github.com/baptistebaquero/ALIDDM/releases/tag/v1.0.4)

**Load models:**
The user has to indicate the path of the folder containing the [trained models for ALI-IOS](https://github.com/baptistebaquero/ALIDDM/releases/tag/v1.0.3).


**Landmark selection:**
For the IOS landmarks, the user has to choose which tooth he need the landmakrs on by checking the label of the tooth on the left table.
Once the folder containing the trained models is loaded. The user can choose the landmark he want to identify with the table on the right showing the available landmarks:
![LM_tab_ios](https://user-images.githubusercontent.com/46842010/180010083-4f7b6e31-edd3-41a2-a696-6a6a1a4d9260.png)



# Acknowledgements

_Authors: Maxime Gillot (University of Michigan), Baptiste Baquero (UoM),Lucia Cevidanes (UoM), Juan Carlos Prieto (UNC)_

Supported by NIDCR R01 024450, AA0F Grabber Family Teaching and Research Award and by Research Enhancement Award Activity 141 from the University of the Pacific, Arthur A. Dugoni School of Dentistry.

# License
This software is licensed under the terms of the [Apache Licence Version 2.0](LICENSE).
