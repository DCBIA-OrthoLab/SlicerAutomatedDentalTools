from enum import Flag, auto
from pathlib import Path

import SegmentEditorEffects
import ctk
import numpy as np
import qt
import slicer
import os
from .IconPath import icon, iconPath
from .PythonDependencyChecker import PythonDependencyChecker, hasInternetConnection
from .Utils import (
    createButton,
    addInCollapsibleLayout,
    set3DViewBackgroundColors,
    setConventionalWideScreenView,
    setBoxAndTextVisibilityOnThreeDViews,
)


class ExportFormat(Flag):
    OBJ = auto()
    STL = auto()
    NIFTI = auto()
    GLTF = auto()


class SegmentationWidget(qt.QWidget):
    def __init__(self, logic=None, parent=None):
        super().__init__(parent)
        self.logic = logic or self._createSlicerSegmentationLogic()
        self._prevSegmentationNode = None
        self._minimumIslandSize_mm3 = 60

        # ================================================
        # Pour le mode folder, on ne sélectionne plus le volume via un Node,
        # mais on utilise la sélection d'un dossier.
        # ================================================
        self.folderPath = ""         # Chemin du dossier sélectionné
        self.folderFiles = []        # Liste des fichiers volume trouvés dans le dossier
        self.currentFileIndex = 0    # Index du fichier en cours de traitement
        self.currentVolumeNode = None  # Volume actuellement chargé

        # Widget de sélection de dossier
        self.folderPathLineEdit = qt.QLineEdit(self)
        self.folderPathLineEdit.setReadOnly(True)
        folderSelectButton = createButton("Select Folder", callback=self.selectFolder)

        # Création d’un widget d’input pour la sélection du dossier
        self.inputWidget = qt.QWidget(self)
        inputLayout = qt.QFormLayout(self.inputWidget)
        inputLayout.setContentsMargins(0, 0, 0, 0)
        inputLayout.addRow("Input Folder:", self.folderPathLineEdit)
        inputLayout.addRow("", folderSelectButton)

        # ================================================
        # Combobox pour le choix du device
        # ================================================
        self.deviceComboBox = qt.QComboBox()
        self.deviceComboBox.addItems(["cuda", "cpu", "mps"])

        # ================================================
        # Nouveau menu déroulant pour sélectionner le modèle
        # ================================================
        self.modelComboBox = qt.QComboBox()
        # Vous pouvez ajouter ici les deux options souhaitées
        self.modelComboBox.addItems(["DentalSegmentator", "PediatricDentalsegmentator"])

        # ================================================
        # Sélecteur de segmentation (inchangé)
        # ================================================
        self.segmentationNodeSelector = slicer.qMRMLNodeComboBox(self)
        self.segmentationNodeSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.segmentationNodeSelector.selectNodeUponCreation = True
        self.segmentationNodeSelector.addEnabled = True
        self.segmentationNodeSelector.removeEnabled = True
        self.segmentationNodeSelector.showHidden = False
        self.segmentationNodeSelector.renameEnabled = True
        self.segmentationNodeSelector.setMRMLScene(slicer.mrmlScene)
        self.segmentationNodeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateSegmentEditorWidget)
        segmentationSelectorComboBox = self.segmentationNodeSelector.findChild("ctkComboBox")
        segmentationSelectorComboBox.defaultText = "Create new Segmentation on Apply"

        # ================================================
        # Création de l'éditeur de segmentation
        # ================================================
        self.segmentEditorWidget = slicer.qMRMLSegmentEditorWidget(self)
        self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        self.segmentEditorWidget.setSegmentationNodeSelectorVisible(False)
        self.segmentEditorWidget.setSourceVolumeNodeSelectorVisible(False)
        self.segmentEditorWidget.layout().setContentsMargins(0, 0, 0, 0)
        self.segmentEditorNode = None

        self.show3DButton = slicer.util.findChild(self.segmentEditorWidget, "Show3DButton")
        smoothingSlider = self.show3DButton.findChild("ctkSliderWidget")
        self.surfaceSmoothingSlider = ctk.ctkSliderWidget(self)
        self.surfaceSmoothingSlider.setToolTip(
            "Higher value means stronger smoothing during closed surface representation conversion."
        )
        self.surfaceSmoothingSlider.decimals = 2
        self.surfaceSmoothingSlider.maximum = 1
        self.surfaceSmoothingSlider.singleStep = 0.1
        self.surfaceSmoothingSlider.setValue(smoothingSlider.value)
        self.surfaceSmoothingSlider.tracking = False
        self.surfaceSmoothingSlider.valueChanged.connect(smoothingSlider.setValue)

        # ================================================
        # Export Widget (inchangé)
        # ================================================
        exportWidget = qt.QWidget()
        exportLayout = qt.QFormLayout(exportWidget)
        self.stlCheckBox = qt.QCheckBox(exportWidget)
        self.stlCheckBox.setChecked(True)
        self.objCheckBox = qt.QCheckBox(exportWidget)
        self.niftiCheckBox = qt.QCheckBox(exportWidget)
        self.gltfCheckBox = qt.QCheckBox(exportWidget)
        self.reductionFactorSlider = ctk.ctkSliderWidget()
        self.reductionFactorSlider.maximum = 1.0
        self.reductionFactorSlider.value = 0.9
        self.reductionFactorSlider.singleStep = 0.01
        self.reductionFactorSlider.toolTip = (
            "Decimation factor determining how much the mesh complexity will be reduced. "
            "Higher value means stronger reduction (smaller files, less details preserved)."
        )
        exportLayout.addRow("Export STL", self.stlCheckBox)
        exportLayout.addRow("Export OBJ", self.objCheckBox)
        exportLayout.addRow("Export NIFTI", self.niftiCheckBox)
        exportLayout.addRow("Export glTF", self.gltfCheckBox)
        exportLayout.addRow("glTF reduction factor :", self.reductionFactorSlider)
        exportLayout.addRow(createButton("Export", callback=self.onExportClicked, parent=exportWidget))

        # ================================================
        # Mise en page principale
        # ================================================
        layout = qt.QVBoxLayout(self)
        self.mainInputWidget = qt.QWidget(self)
        mainInputLayout = qt.QFormLayout(self.mainInputWidget)
        mainInputLayout.setContentsMargins(0, 0, 0, 0)
        mainInputLayout.addRow(self.inputWidget)                   # Sélection du dossier
        mainInputLayout.addRow(self.segmentationNodeSelector)
        mainInputLayout.addRow("Device:", self.deviceComboBox)
        # Ajout du menu déroulant pour le modèle
        mainInputLayout.addRow("Model:", self.modelComboBox)
        layout.addWidget(self.mainInputWidget)

        self.applyButton = createButton(
            "Apply",
            callback=self.onApplyClicked,
            toolTip="Click to run the segmentation.",
            icon=icon("start_icon.png")
        )

        self.currentInfoTextEdit = qt.QTextEdit()
        self.currentInfoTextEdit.setReadOnly(True)
        self.currentInfoTextEdit.setLineWrapMode(qt.QTextEdit.NoWrap)
        self.fullInfoLogs = []
        self.stopWidget = qt.QVBoxLayout()

        self.stopButton = createButton(
            "Stop",
            callback=self.onStopClicked,
            toolTip="Click to Stop the segmentation."
        )
        self.stopWidgetContainer = qt.QWidget(self)
        stopLayout = qt.QVBoxLayout(self.stopWidgetContainer)
        stopLayout.setContentsMargins(0, 0, 0, 0)
        stopLayout.addWidget(self.stopButton)
        stopLayout.addWidget(self.currentInfoTextEdit)
        self.stopWidgetContainer.setVisible(False)

        self.loading = qt.QMovie(iconPath("loading.gif"))
        self.loading.setScaledSize(qt.QSize(24, 24))
        self.loading.frameChanged.connect(self._updateStopIcon)
        self.loading.start()

        self.applyWidget = qt.QWidget(self)
        applyLayout = qt.QHBoxLayout(self.applyWidget)
        applyLayout.setContentsMargins(0, 0, 0, 0)
        applyLayout.addWidget(self.applyButton, 1)
        applyLayout.addWidget(
            createButton("", callback=self.showInfoLogs, icon=icon("info.png"), toolTip="Show logs.")
        )

        layout.addWidget(self.applyWidget)
        layout.addWidget(self.stopWidgetContainer)
        layout.addWidget(self.segmentEditorWidget)

        surfaceSmoothingLayout = qt.QFormLayout()
        surfaceSmoothingLayout.setContentsMargins(0, 0, 0, 0)
        surfaceSmoothingLayout.addRow("Surface smoothing :", self.surfaceSmoothingSlider)
        layout.addLayout(surfaceSmoothingLayout)
        layout.addWidget(exportWidget)
        addInCollapsibleLayout(exportWidget, layout, "Export segmentation", isCollapsed=False)
        layout.addStretch()

        self.isStopping = False

        self._dependencyChecker = PythonDependencyChecker()
        self.processedVolumes = {}

        # Pour le mode folder, aucun volume n'est sélectionné manuellement.
        self.onInputChangedForLoadedVolume(None)
        self.updateSegmentEditorWidget()
        self.sceneCloseObserver = slicer.mrmlScene.AddObserver(slicer.mrmlScene.EndCloseEvent, self.onSceneChanged)
        self.onSceneChanged(doStopInference=False)
        self._connectSegmentationLogic()

    def __del__(self):
        slicer.mrmlScene.RemoveObserver(self.sceneCloseObserver)
        super().__del__()

    def selectFolder(self):
        folderPath = qt.QFileDialog.getExistingDirectory(self, "Select Folder Containing Volumes")
        if folderPath:
            self.folderPath = folderPath
            self.folderPathLineEdit.text = folderPath
            folder = Path(folderPath)
            # Filtrer ici selon vos formats, par exemple tous les fichiers NIfTI
            self.folderFiles = list(folder.glob("*.nii*")) + list(folder.glob("*.gipl")) + list(folder.glob("*.gipl.gz"))

            self.currentFileIndex = 0
            self.onProgressInfo(f"Found {len(self.folderFiles)} file(s) in the folder.")

    def onSceneChanged(self, *_, doStopInference=True):
        if doStopInference:
            self.onStopClicked()
        self.segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        self.segmentEditorWidget.setMRMLSegmentEditorNode(self.segmentEditorNode)
        self.processedVolumes = {}
        self._prevSegmentationNode = None
        self._initSlicerDisplay()

    @staticmethod
    def _initSlicerDisplay():
        set3DViewBackgroundColors([1, 1, 1], [1, 1, 1])
        setConventionalWideScreenView()
        setBoxAndTextVisibilityOnThreeDViews(False)

    def _updateStopIcon(self):
        self.stopButton.setIcon(qt.QIcon(self.loading.currentPixmap()))

    def onStopClicked(self):
        self.isStopping = True
        self.logic.stopSegmentation()
        self.logic.waitForSegmentationFinished()
        slicer.app.processEvents()
        self.isStopping = False
        self._setApplyVisible(True)

    def onApplyClicked(self, *_):
        """
        Pour le mode folder, on lance le traitement sur l'ensemble des fichiers présents dans le dossier sélectionné.
        """
        if not self.folderPath:
            slicer.util.errorDisplay("Veuillez sélectionner un dossier contenant des volumes.")
            return

        self.currentInfoTextEdit.clear()
        self._setApplyVisible(False)

        if not self.folderFiles:
            slicer.util.errorDisplay("Aucun fichier volume valide trouvé dans le dossier.")
            self._setApplyVisible(True)
            return

        if not self.isNNUNetModuleInstalled() or self.logic is None:
            slicer.util.errorDisplay(
                "This module depends on the NNUNet module. Please install the NNUNet module and restart to proceed."
            )
            return

        if not self._installNNUNetIfNeeded():
            self._setApplyVisible(True)
            return

        if not self._dependencyChecker.downloadWeightsIfNeeded(self.onProgressInfo):
            self._setApplyVisible(True)
            return

        # Lancer le traitement sur le premier fichier du dossier
        self.processNextFile()

    def processNextFile(self):
        if self.currentFileIndex >= len(self.folderFiles):
            self.onProgressInfo("Tous les fichiers du dossier ont été traités.")
            self._setApplyVisible(True)
            return

        filePath = self.folderFiles[self.currentFileIndex]
        self.onProgressInfo(f"Traitement du fichier : {filePath}")

        loadedVolume = slicer.util.loadVolume(str(filePath))
        if not loadedVolume:
            self.onProgressInfo(f"Erreur lors du chargement de {filePath}. Passage au suivant.")
            self.currentFileIndex += 1
            self.processNextFile()
            return

        self.currentVolumeNode = loadedVolume
        self.onInputChangedForLoadedVolume(loadedVolume)
        self.onApplyClickedForVolume(loadedVolume)

    def onInputChangedForLoadedVolume(self, volumeNode):
        if volumeNode:
            slicer.util.setSliceViewerLayers(background=volumeNode)
            slicer.util.resetSliceViews()
            self._restoreProcessedSegmentationForVolume(volumeNode)

    def _restoreProcessedSegmentationForVolume(self, volumeNode):
        segmentationNode = self.processedVolumes.get(volumeNode)
        self.segmentationNodeSelector.setCurrentNode(segmentationNode)

    def onApplyClickedForVolume(self, volumeNode):
        from SlicerNNUNetLib import Parameter
        selectedModel = self.modelComboBox.currentText
        if selectedModel == "PediatricDentalsegmentator":
            print("Selected Model : ",selectedModel)
            # Chemin de base où le modèle complet doit être installé
            basePath = Path(__file__).parent.joinpath("..", "Resources", "ML", "Dataset001_380CT", "nnUNetTrainer__nnUNetPlans__3d_fullres").resolve()
            # On choisit fold_0 (vous pouvez adapter par fold_1 si nécessaire)
            fold_path = basePath.joinpath("fold_0")
            if not fold_path.exists():
                fold_path.mkdir(parents=True, exist_ok=True)
            # Chemin du checkpoint dans le dossier fold_0
            pediatricCheckpoint = fold_path.joinpath("checkpoint_final.pth")
            # Si le checkpoint n'existe pas, télécharger le checkpoint ainsi que dataset.json et plans.json dans le répertoire basePath
            if not pediatricCheckpoint.exists():
                url_checkpoint = "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/PEDIATRICDENTALSEG_MODEL/checkpoint_final.pth"
                url_dataset = "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/PEDIATRICDENTALSEG_MODEL/dataset.json"
                url_plans = "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/PEDIATRICDENTALSEG_MODEL/plans.json"
                self.onProgressInfo("Téléchargement du modèle pediatricdentalseg...")
                # Télécharger le checkpoint ; on convertit le Path en string pour downloadFile
                slicer.util.downloadFile(url_checkpoint, str(pediatricCheckpoint))
                # Télécharger dataset.json et plans.json dans basePath
                slicer.util.downloadFile(url_dataset, str(basePath.joinpath("dataset.json")))
                slicer.util.downloadFile(url_plans, str(basePath.joinpath("plans.json")))
            # Pour nnUNet, le modelPath doit pointer sur le dossier contenant dataset.json et fold_x
            parameter = Parameter(folds="0", modelPath=basePath, device=self.deviceComboBox.currentText)


########################################################################  AVAILABLE SOON ###############################################################################################################


        # if selectedModel == "UniversalLabDentalsegmentator":
        #     print("Selected Model : ",selectedModel)
        #     # Chemin de base où le modèle complet doit être installé
        #     basePath = Path(__file__).parent.joinpath("..", "Resources", "ML", "Dataset002_380CT", "nnUNetTrainer__nnUNetPlans__3d_fullres").resolve()
        #     # On choisit fold_0 (vous pouvez adapter par fold_1 si nécessaire)
        #     fold_path = basePath.joinpath("fold_0")
        #     if not fold_path.exists():
        #         fold_path.mkdir(parents=True, exist_ok=True)
        #     # Chemin du checkpoint dans le dossier fold_0
        #     pediatricCheckpoint = fold_path.joinpath("checkpoint_final.pth")
        #     # Si le checkpoint n'existe pas, télécharger le checkpoint ainsi que dataset.json et plans.json dans le répertoire basePath
        #     if not pediatricCheckpoint.exists():
        #         url_checkpoint = "https://github.com/ashmoy/tooth_seg/releases/download/model_ULab/checkpoint_final.pth"
        #         url_dataset = "https://github.com/ashmoy/tooth_seg/releases/download/model_ULab/dataset.json"
        #         url_plans = "https://github.com/ashmoy/tooth_seg/releases/download/model_ULab/plans.json"
        #         self.onProgressInfo("Téléchargement du modèle pediatricdentalseg...")
        #         # Télécharger le checkpoint ; on convertit le Path en string pour downloadFile
        #         slicer.util.downloadFile(url_checkpoint, str(pediatricCheckpoint))
        #         # Télécharger dataset.json et plans.json dans basePath
        #         slicer.util.downloadFile(url_dataset, str(basePath.joinpath("dataset.json")))
        #         slicer.util.downloadFile(url_plans, str(basePath.joinpath("plans.json")))
        #     # Pour nnUNet, le modelPath doit pointer sur le dossier contenant dataset.json et fold_x
        #     parameter = Parameter(folds="0", modelPath=basePath, device=self.deviceComboBox.currentText)


#########################################################################################################################################################################################################

        else:
            print("Selected Model : ",selectedModel)
            parameter = Parameter(folds="0", modelPath=self.nnUnetFolder(), device=self.deviceComboBox.currentText)
                
        if not parameter.isSelectedDeviceAvailable():
            deviceName = parameter.device.upper()
            ret = qt.QMessageBox.question(
                self,
                f"{deviceName} device not available",
                f"Selected device ({deviceName}) is not available and will default to CPU.\n"
                "Running the segmentation may take up to 1 hour.\n"
                "Would you like to proceed?"
            )
            if ret == qt.QMessageBox.No:
                self._setApplyVisible(True)
                return
        slicer.app.processEvents()
        self.logic.setParameter(parameter)
        self.logic.startSegmentation(volumeNode)


    def onInferenceFinished(self, *_):
        if self.isStopping:
            self._setApplyVisible(True)
            return

        try:
            self.onProgressInfo("Loading inference results...")
            self._loadSegmentationResults()
            self.onProgressInfo("Inference ended successfully.")
        except RuntimeError as e:
            slicer.util.errorDisplay(e)
            self.onProgressInfo(f"Error loading results :\n{e}")
        finally:
            if self.folderPath:
                self.currentFileIndex += 1
                if self.currentFileIndex < len(self.folderFiles):
                    self.processNextFile()
                else:
                    self.onProgressInfo("Tous les fichiers du dossier ont été traités.")
                    self._setApplyVisible(True)
            else:
                self._setApplyVisible(True)

    def _loadSegmentationResults(self):
        currentSegmentation = self.getCurrentSegmentationNode()
        segmentationNode = self.logic.loadSegmentation()
        segmentationNode.SetName(self.currentVolumeNode.GetName() + "_Segmentation")
        if currentSegmentation is not None:
            self._copySegmentationResultsToExistingNode(currentSegmentation, segmentationNode)
        else:
            self.segmentationNodeSelector.setCurrentNode(segmentationNode)
        slicer.app.processEvents()
        self._updateSegmentationDisplay()
        # self._postProcessSegments()
        self._storeProcessedSegmentation()

    @staticmethod
    def _copySegmentationResultsToExistingNode(currentSegmentation, segmentationNode):
        currentName = currentSegmentation.GetName()
        currentSegmentation.Copy(segmentationNode)
        currentSegmentation.SetName(currentName)
        slicer.mrmlScene.RemoveNode(segmentationNode)

    @staticmethod
    def toRGB(colorString):
        color = qt.QColor(colorString)
        return color.redF(), color.greenF(), color.blueF()

    def _updateSegmentationDisplay(self):
        segmentationNode = self.getCurrentSegmentationNode()
        if not segmentationNode:
            return
        self._initializeSegmentationNodeDisplay(segmentationNode)
        segmentation = segmentationNode.GetSegmentation()
        selectedModel = self.modelComboBox.currentText
################################################################################# AVAILABLE SOON #####################################################################################################
       
        # if selectedModel == "UniversalLabDentalsegmentator":
        #                 # Pour le modèle UniversalLabDentalsegmentator,
        #     # on considère 55 labels (on ignore "background")
        #     UNIVERSAL_LABELS = [
        #         "Upper-right third molar",
        #         "Upper-right second molar",
        #         "Upper-right first molar",
        #         "Upper-right second premolar",
        #         "Upper-right first premolar",
        #         "Upper-right canine",
        #         "Upper-right lateral incisor",
        #         "Upper-right central incisor",
        #         "Upper-left central incisor",
        #         "Upper-left lateral incisor",
        #         "Upper-left canine",
        #         "Upper-left first premolar",
        #         "Upper-left second premolar",
        #         "Upper-left first molar",
        #         "Upper-left second molar",
        #         "Upper-left third molar",
        #         "Lower-left third molar",
        #         "Lower-left second molar",
        #         "Lower-left first molar",
        #         "Lower-left second premolar",
        #         "Lower-left first premolar",
        #         "Lower-left canine",
        #         "Lower-left lateral incisor",
        #         "Lower-left central incisor",
        #         "Lower-right central incisor",
        #         "Lower-right lateral incisor",
        #         "Lower-right canine",
        #         "Lower-right first premolar",
        #         "Lower-right second premolar",
        #         "Lower-right first molar",
        #         "Lower-right second molar",
        #         "Lower-right third molar",
        #         "Upper-right second molar (baby)",
        #         "Upper-right first molar (baby)",
        #         "Upper-right canine (baby)",
        #         "Upper-right lateral incisor (baby)",
        #         "Upper-right central incisor (baby)",
        #         "Upper-left central incisor (baby)",
        #         "Upper-left lateral incisor (baby)",
        #         "Upper-left canine (baby)",
        #         "Upper-left first molar (baby)",
        #         "Upper-left second molar (baby)",
        #         "Lower-left second molar (baby)",
        #         "Lower-left first molar (baby)",
        #         "Lower-left canine (baby)",
        #         "Lower-left lateral incisor (baby)",
        #         "Lower-left central incisor (baby)",
        #         "Lower-right central incisor (baby)",
        #         "Lower-right lateral incisor (baby)",
        #         "Lower-right canine (baby)",
        #         "Lower-right first molar (baby)",
        #         "Lower-right second molar (baby)",
        #         "Mandible",
        #         "Maxilla",
        #         "Mandibular canal"
        #     ]

        #     # Une palette de 55 couleurs en hexadécimal (vous pouvez adapter les codes)
        #     UNIVERSAL_COLORS = [
        #         "#FF0000",  # Upper-right third molar
        #         "#00FF00",  # Upper-right second molar
        #         "#0000FF",  # Upper-right first molar
        #         "#FFFF00",  # Upper-right second premolar
        #         "#FF00FF",  # Upper-right first premolar
        #         "#00FFFF",  # Upper-right canine
        #         "#800000",  # Upper-right lateral incisor
        #         "#008000",  # Upper-right central incisor
        #         "#000080",  # Upper-left central incisor
        #         "#808000",  # Upper-left lateral incisor
        #         "#800080",  # Upper-left canine
        #         "#008080",  # Upper-left first premolar
        #         "#C0C0C0",  # Upper-left second premolar
        #         "#808080",  # Upper-left first molar
        #         "#FFA500",  # Upper-left second molar
        #         "#F0E68C",  # Upper-left third molar
        #         "#B22222",  # Lower-left third molar
        #         "#8FBC8F",  # Lower-left second molar
        #         "#483D8B",  # Lower-left first molar
        #         "#2F4F4F",  # Lower-left second premolar
        #         "#00CED1",  # Lower-left first premolar
        #         "#9400D3",  # Lower-left canine
        #         "#FF1493",  # Lower-left lateral incisor
        #         "#7FFF00",  # Lower-left central incisor
        #         "#1E90FF",  # Lower-right central incisor
        #         "#FF4500",  # Lower-right lateral incisor
        #         "#DA70D6",  # Lower-right canine
        #         "#EEE8AA",  # Lower-right first premolar
        #         "#98FB98",  # Lower-right second premolar
        #         "#AFEEEE",  # Lower-right first molar
        #         "#DB7093",  # Lower-right second molar
        #         "#FFE4E1",  # Lower-right third molar
        #         "#FFDAB9",  # Upper-right second molar (baby)
        #         "#CD5C5C",  # Upper-right first molar (baby)
        #         "#F08080",  # Upper-right canine (baby)
        #         "#E9967A",  # Upper-right lateral incisor (baby)
        #         "#FA8072",  # Upper-right central incisor (baby)
        #         "#FF7F50",  # Upper-left central incisor (baby)
        #         "#FF6347",  # Upper-left lateral incisor (baby)
        #         "#00FA9A",  # Upper-left canine (baby)
        #         "#00FF7F",  # Upper-left first molar (baby)
        #         "#4682B4",  # Upper-left second molar (baby)
        #         "#87CEEB",  # Lower-left second molar (baby)
        #         "#6A5ACD",  # Lower-left first molar (baby)
        #         "#7B68EE",  # Lower-left canine (baby)
        #         "#4169E1",  # Lower-left lateral incisor (baby)
        #         "#6495ED",  # Lower-left central incisor (baby)
        #         "#B0C4DE",  # Lower-right central incisor (baby)
        #         "#008080",  # Lower-right lateral incisor (baby)
        #         "#ADFF2F",  # Lower-right canine (baby)
        #         "#FF69B4",  # Lower-right first molar (baby)
        #         "#CD853F",  # Lower-right second molar (baby)
        #         "#D2691E",  # Mandible
        #         "#B8860B",  # Maxilla
        #         "#A0522D"   # Mandibular canal
        #     ]

        #     # Pour une opacité uniforme, par exemple à 1.0 pour chaque segment
        #     UNIVERSAL_OPACITIES = [
        #         1.0,  # Upper-right third molar
        #         1.0,  # Upper-right second molar
        #         1.0,  # Upper-right first molar
        #         1.0,  # Upper-right second premolar
        #         1.0,  # Upper-right first premolar
        #         1.0,  # Upper-right canine
        #         1.0,  # Upper-right lateral incisor
        #         1.0,  # Upper-right central incisor
        #         1.0,  # Upper-left central incisor
        #         1.0,  # Upper-left lateral incisor
        #         1.0,  # Upper-left canine
        #         1.0,  # Upper-left first premolar
        #         1.0,  # Upper-left second premolar
        #         1.0,  # Upper-left first molar
        #         1.0,  # Upper-left second molar
        #         1.0,  # Upper-left third molar
        #         1.0,  # Lower-left third molar
        #         1.0,  # Lower-left second molar
        #         1.0,  # Lower-left first molar
        #         1.0,  # Lower-left second premolar
        #         1.0,  # Lower-left first premolar
        #         1.0,  # Lower-left canine
        #         1.0,  # Lower-left lateral incisor
        #         1.0,  # Lower-left central incisor
        #         1.0,  # Lower-right central incisor
        #         1.0,  # Lower-right lateral incisor
        #         1.0,  # Lower-right canine
        #         1.0,  # Lower-right first premolar
        #         1.0,  # Lower-right second premolar
        #         1.0,  # Lower-right first molar
        #         1.0,  # Lower-right second molar
        #         1.0,  # Lower-right third molar
        #         1.0,  # Upper-right second molar (baby)
        #         1.0,  # Upper-right first molar (baby)
        #         1.0,  # Upper-right canine (baby)
        #         1.0,  # Upper-right lateral incisor (baby)
        #         1.0,  # Upper-right central incisor (baby)
        #         1.0,  # Upper-left central incisor (baby)
        #         1.0,  # Upper-left lateral incisor (baby)
        #         1.0,  # Upper-left canine (baby)
        #         1.0,  # Upper-left first molar (baby)
        #         1.0,  # Upper-left second molar (baby)
        #         1.0,  # Lower-left second molar (baby)
        #         1.0,  # Lower-left first molar (baby)
        #         1.0,  # Lower-left canine (baby)
        #         1.0,  # Lower-left lateral incisor (baby)
        #         1.0,  # Lower-left central incisor (baby)
        #         1.0,  # Lower-right central incisor (baby)
        #         1.0,  # Lower-right lateral incisor (baby)
        #         1.0,  # Lower-right canine (baby)
        #         1.0,  # Lower-right first molar (baby)
        #         1.0,  # Lower-right second molar (baby)
        #         0.45,  # Mandible
        #         0.45,  # Maxilla
        #         0.45   # Mandibular canal
        #     ]
        #     labels = UNIVERSAL_LABELS
        #     colors = UNIVERSAL_COLORS
        #     opacities = UNIVERSAL_OPACITIES
        #     # On crée des identifiants de segment de la même manière qu'avant, par exemple "Segment_1", "Segment_2", ...
        #     segmentIds = [f"Segment_{i+1}" for i in range(len(labels))]
        #     segmentationDisplayNode = segmentationNode.GetDisplayNode()
        #     for segmentId, label, color, opacity in zip(segmentIds, labels, colors, opacities):
        #         segment = segmentation.GetSegment(segmentId)
        #         if segment is None:
        #             continue
        #         segment.SetName(label)
        #         segment.SetColor(*self.toRGB(color))
        #         segmentationDisplayNode.SetSegmentOpacity3D(segmentId, opacity)
       # else:
########################################################################################################################################################################################################


        labels = ["Maxilla & Upper Skull", "Mandible", "Upper Teeth", "Lower Teeth", "Mandibular canal"]
        colors = [self.toRGB(c) for c in ["#E3DD90", "#D4A1E6", "#DC9565", "#EBDFB4", "#D8654F"]]
        opacities = [0.45, 0.45, 1.0, 1.0, 1.0]
        segmentIds = [f"Segment_{i + 1}" for i in range(len(labels))]
        segmentationDisplayNode = self.getCurrentSegmentationNode().GetDisplayNode()
        for segmentId, label, color, opacity in zip(segmentIds, labels, colors, opacities):
            segment = segmentation.GetSegment(segmentId)
            if segment is None:
                continue
            segment.SetName(label)
            segment.SetColor(*color)
            segmentationDisplayNode.SetSegmentOpacity3D(segmentId, opacity)
        self.show3DButton.setChecked(True)
        slicer.util.resetThreeDViews()

    def _initializeSegmentationNodeDisplay(self, segmentationNode):
        if not segmentationNode:
            return
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(self.getCurrentVolumeNode())
        if not segmentationNode.GetDisplayNode():
            segmentationNode.CreateDefaultDisplayNodes()
            slicer.app.processEvents()
        segmentationNode.SetDisplayVisibility(True)
        layoutManager = slicer.app.layoutManager()
        threeDWidget = layoutManager.threeDWidget(0)
        threeDWidget.threeDView().rotateToViewAxis(3)
        slicer.util.resetThreeDViews()

    def _postProcessSegments(self):
        self.onProgressInfo("Post processing results...")
        self._keepLargestIsland("Segment_1")
        self._removeSmallIsland("Segment_3")
        self._removeSmallIsland("Segment_4")
        self.onProgressInfo("Post processing done.")

    def _keepLargestIsland(self, segmentId):
        segment = self._getSegment(segmentId)
        if not segment:
            return
        self.onProgressInfo(f"Keep largest region for {segment.GetName()}...")
        self.segmentEditorWidget.setCurrentSegmentID(segmentId)
        effect = self.segmentEditorWidget.effectByName("Islands")
        effect.setParameter("Operation", SegmentEditorEffects.KEEP_LARGEST_ISLAND)
        effect.self().onApply()

    def _removeSmallIsland(self, segmentId):
        segment = self._getSegment(segmentId)
        if not segment:
            return
        self.onProgressInfo(f"Remove small voxels for {segment.GetName()}...")
        self.segmentEditorWidget.setCurrentSegmentID(segmentId)
        voxelSize_mm3 = np.cumprod(self.getCurrentVolumeNode().GetSpacing())[-1]
        minimumIslandSize = int(np.ceil(self._minimumIslandSize_mm3 / voxelSize_mm3))
        effect = self.segmentEditorWidget.effectByName("Islands")
        effect.setParameter("Operation", SegmentEditorEffects.REMOVE_SMALL_ISLANDS)
        effect.setParameter("MinimumSize", minimumIslandSize)
        effect.self().onApply()

    def _getSegment(self, segmentId):
        segmentationNode = self.getCurrentSegmentationNode()
        if not segmentationNode:
            return
        return segmentationNode.GetSegmentation().GetSegment(segmentId)

    def onInferenceError(self, errorMsg):
        if self.isStopping:
            return
        self._setApplyVisible(True)
        slicer.util.errorDisplay("Encountered error during inference :\n" + str(errorMsg))


    def onProgressInfo(self, infoMsg):
        infoMsg = self.removeImageIOError(infoMsg)
        self.currentInfoTextEdit.insertPlainText(infoMsg + "\n")
        self.moveTextEditToEnd(self.currentInfoTextEdit)
        self.insertDatedInfoLogs(infoMsg)
        slicer.app.processEvents()

    @staticmethod
    def removeImageIOError(infoMsg):
        return "\n".join([msg for msg in infoMsg.strip().splitlines() if "Error ImageIO factory" not in msg])

    def insertDatedInfoLogs(self, infoMsg):
        now = qt.QDateTime.currentDateTime().toString("yyyy/MM/dd hh:mm:ss.zzz")
        self.fullInfoLogs.extend([f"{now} :: {msgLine}" for msgLine in infoMsg.splitlines()])

    def showInfoLogs(self):
        dialog = qt.QDialog()
        layout = qt.QVBoxLayout(dialog)
        textEdit = qt.QTextEdit()
        textEdit.setReadOnly(True)
        textEdit.append("\n".join(self.fullInfoLogs))
        textEdit.setLineWrapMode(qt.QTextEdit.NoWrap)
        self.moveTextEditToEnd(textEdit)
        layout.addWidget(textEdit)
        dialog.setWindowFlags(qt.Qt.WindowCloseButtonHint)
        dialog.resize(slicer.util.mainWindow().size * 0.7)
        dialog.exec()

    @staticmethod
    def moveTextEditToEnd(textEdit):
        textEdit.verticalScrollBar().setValue(textEdit.verticalScrollBar().maximum)

    def _setApplyVisible(self, isVisible):
        self.applyWidget.setVisible(isVisible)
        self.stopWidgetContainer.setVisible(not isVisible)
        self.inputWidget.setEnabled(isVisible)

    def getCurrentVolumeNode(self):
        return self.currentVolumeNode

    def getCurrentSegmentationNode(self):
        return self.segmentationNodeSelector.currentNode()

    def _storeProcessedSegmentation(self):
        volumeNode = self.getCurrentVolumeNode()
        segmentationNode = self.getCurrentSegmentationNode()
        if volumeNode and segmentationNode:
            self.processedVolumes[volumeNode] = segmentationNode
    def updateSegmentEditorWidget(self, *_):
        """
        Met à jour le widget d'édition de segmentation en fonction du nœud actuellement sélectionné.
        """
        if self._prevSegmentationNode:
            self._prevSegmentationNode.SetDisplayVisibility(False)

        segmentationNode = self.getCurrentSegmentationNode()
        self._prevSegmentationNode = segmentationNode
        self._initializeSegmentationNodeDisplay(segmentationNode)
        self.segmentEditorWidget.setSegmentationNode(segmentationNode)
        self.segmentEditorWidget.setSourceVolumeNode(self.getCurrentVolumeNode())
    def getSelectedExportFormats(self):
        selectedFormats = ExportFormat(0)
        checkBoxes = {
            self.objCheckBox: ExportFormat.OBJ,
            self.stlCheckBox: ExportFormat.STL,
            self.niftiCheckBox: ExportFormat.NIFTI,
            self.gltfCheckBox: ExportFormat.GLTF
        }
        for checkBox, exportFormat in checkBoxes.items():
            if checkBox.isChecked():
                selectedFormats |= exportFormat
        return selectedFormats

    def onExportClicked(self):
        segmentationNode = self.getCurrentSegmentationNode()
        if not segmentationNode:
            slicer.util.warningDisplay("Please select a valid segmentation before exporting.")
            return

        selectedFormats = self.getSelectedExportFormats()
        if selectedFormats == ExportFormat(0):
            slicer.util.warningDisplay("Please select at least one export format before exporting.")
            return

        folderPath = qt.QFileDialog.getExistingDirectory(self, "Please select the export folder")
        if not folderPath:
            return

        with slicer.util.tryWithErrorDisplay(f"Export to {folderPath} failed.", waitCursor=True):
            self.exportSegmentation(segmentationNode, folderPath, selectedFormats)
            slicer.util.infoDisplay(f"Export successful to {folderPath}.")

    def exportSegmentation(self, segmentationNode, folderPath, selectedFormats):
        for closedSurfaceExport in [ExportFormat.STL, ExportFormat.OBJ]:
            if selectedFormats & closedSurfaceExport:
                slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsClosedSurfaceRepresentationToFiles(
                    folderPath,
                    segmentationNode,
                    None,
                    closedSurfaceExport.name,
                    True,
                    1.0,
                    False
                )

        if selectedFormats & ExportFormat.NIFTI:
            slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsBinaryLabelmapRepresentationToFiles(
                folderPath,
                segmentationNode,
                None,
                "nii.gz"
            )

        if selectedFormats & ExportFormat.GLTF:
            self._exportToGLTF(segmentationNode, folderPath)

    def _exportToGLTF(self, segmentationNode, folderPath, tryInstall=True):
        try:
            from OpenAnatomyExport import OpenAnatomyExportLogic
            logic = OpenAnatomyExportLogic()
            shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
            segmentationItem = shNode.GetItemByDataNode(self.segmentationNodeSelector.currentNode())
            logic.exportModel(segmentationItem, folderPath, self.reductionFactorSlider.value, "glTF")
        except ImportError:
            if not tryInstall or not hasInternetConnection():
                slicer.util.errorDisplay(
                    "Failed to export to glTF. Try installing the SlicerOpenAnatomy extension manually to continue."
                )
                return
            self._installOpenAnatomyExtension()
            self._exportToGLTF(segmentationNode, folderPath, tryInstall=False)

    @classmethod
    def _installOpenAnatomyExtension(cls):
        extensionManager = slicer.app.extensionsManagerModel()
        extensionManager.setInteractive(False)
        extName = "SlicerOpenAnatomy"
        if extensionManager.isExtensionInstalled(extName):
            return

        success = extensionManager.installExtensionFromServer(extName, False, False)
        if not success:
            return

        moduleName = "OpenAnatomyExport"
        modulePath = extensionManager.extensionModulePaths(extName)[0] + f"/{moduleName}.py"
        factory = slicer.app.moduleManager().factoryManager()
        factory.registerModule(qt.QFileInfo(modulePath))
        factory.loadModules([moduleName])

    @staticmethod
    def isNNUNetModuleInstalled():
        try:
            import SlicerNNUNetLib
            return True
        except ImportError:
            return False

    def _installNNUNetIfNeeded(self) -> bool:
        from SlicerNNUNetLib import InstallLogic
        logic = InstallLogic()
        logic.progressInfo.connect(self.onProgressInfo)
        return logic.setupPythonRequirements()

    def _createSlicerSegmentationLogic(self):
        if not self.isNNUNetModuleInstalled():
            return None
        from SlicerNNUNetLib import SegmentationLogic
        return SegmentationLogic()

    def _connectSegmentationLogic(self):
        if self.logic is None:
            return
        self.logic.progressInfo.connect(self.onProgressInfo)
        self.logic.errorOccurred.connect(self.onInferenceError)
        self.logic.inferenceFinished.connect(self.onInferenceFinished)

    @classmethod
    def nnUnetFolder(cls) -> Path:
        fileDir = Path(__file__).parent
        return fileDir.joinpath("..", "Resources", "ML").resolve()
