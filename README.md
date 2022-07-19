# Slicer Automated Dental Tools 

_Authors: Maxime Gillot (University of Michigan), Baptiste Baquero (UoM)_


Slicer automated dental tools is an extension perform automatic **segmentation** and **landmark identification** using machine learning tools.
This extension will allow you to :
- segment CBCT scan using [AMASSS](https://github.com/Maxlo24/AMASSS_CBCT)
- Identify landmarks in CBCT using [ALI-CBCT](https://github.com/Maxlo24/ALI_CBCT)
- Identify landmarks in IOS using [ALI-IOS](https://github.com/baptistebaquero/ALIDDM)

<img src="SlicerAutomaticTools.png" alt="Extension Logo" width="100"/>

## Requirements 

```
Slicer 5.1.0 or later
```

### :warning: Warning

- All modules relies on machine learning tools that **requires GPU** for optimum use
- The user will have to **download the trained networks**  required for each module




---
# How to use the extension

All modules can work with one file or a whole semple as input.


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
The user can choose the structure to segment using the selection tab.
Depending on the type of CBCT to segment, the user can check the "Use small FOV models" for higher definition.


## ALI Module
<img src="ALI/Resources/Icons/ALI.png" alt="Extension Logo" width="50"/>

ALI module will allow you to identify landmarks on CBCT or IOS scan using [ALI-CBCT](https://github.com/Maxlo24/ALI_CBCT) or [ALI-IOS](https://github.com/baptistebaquero/ALIDDM) algortihm.



- #### ALI-CBCT
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

- #### ALI-IOS

**Input file:**
The input has to be an oriented IOS segmented with the [Universal Numbering System](https://en.wikipedia.org/wiki/Universal_Numbering_System).
This segmentation can be automatically done using the [SlicerJawSegmentation](https://github.com/MathieuLeclercq/SlicerJawSegmentation) extention.
The input can be a single IOS loaded on slicer or a folder containg IOS with the following extention:
```
.vtk
```

**Load models:**
The user has to indicate the path of the folder containing the [trained models for ALI-IOS](https://github.com/baptistebaquero/ALIDDM/releases/tag/v1.0.3).


