import logging
import os
from typing import Annotated
import vtk
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (parameterNodeWrapper,WithinRange,)
from slicer import vtkMRMLScalarVolumeNode

# ------------------------------------------------------------------------------------------------
# LIB
# ------------------------------------------------------------------------------------------------
def check_lib_installed(lib_name, required_version=None):
    pass #TO DO

def install_function(self,list_libs:list):
    pass #TO DO


class CNE_UI(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("CNE_UI")
        self.parent.categories = ["Automated Dental Tools"]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Paul Dumont, University of North Carolina, Chapell Hill"]  
        self.parent.helpText = _("""
        This tool helps to create summaries of clinical notes. 
        See more information in <a href="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools">documentation</a>.
        """)
        self.parent.acknowledgementText = _("""
        This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
        and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
        """)

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#
def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # CNE_UI1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="CNE_UI",
        sampleName="CNE_UI1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "CNE_UI1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="CNE_UI1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="CNE_UI1",
    )

    # CNE_UI2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="CNE_UI",
        sampleName="CNE_UI2",
        thumbnailFileName=os.path.join(iconsPath, "CNE_UI2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="CNE_UI2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="CNE_UI2",
    )


#
# CNE_UIParameterNode
#
@parameterNodeWrapper
class CNE_UIParameterNode:
    """
    Parameters for Clinical Notes Extraction UI.

    notesFolder_input - Folder containing clinical notes (.docx/.pdf/.txt).
    modelType - Model type selection: 'Mini' (Light/Fast) or 'Max' (Heavy/Precise).
    notesType - Notes type selection: 'TMJ' or 'Ortho'.
    notesFolder_output - Folder for summary output.
    """

    notesFolder_input: str = "" # Path to input notes folder
    modelType: str = ""  # "Mini" or "Max"
    notesType: str = ""   # "TMJ" or "Ortho"
    notesFolder_output: str = ""  # Path to output summary folder

#
# CNE_UIWidget
#
class CNE_UIWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)


        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/CNE_UI.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Create logic class.
        self.logic = CNE_UILogic()

        # Create QButtonGroup for model type selection
        import qt
        self.modelTypeButtonGroup = qt.QButtonGroup()
        self.modelTypeButtonGroup.addButton(self.ui.modelQuickRadioButton)
        self.modelTypeButtonGroup.addButton(self.ui.modelProRadioButton)
        self.modelTypeButtonGroup.setExclusive(True)

        # Create QButtonGroup for notes type selection
        self.notesTypeButtonGroup = qt.QButtonGroup()
        self.notesTypeButtonGroup.addButton(self.ui.notesTypeTMJRadioButton)
        self.notesTypeButtonGroup.addButton(self.ui.notesTypeOrthoRadioButton)
        self.notesTypeButtonGroup.setExclusive(True)

        # ==========================================================
        # ---> AJOUTER ICI : Création de la barre de progression <--
        # ==========================================================
        self.cliProgressBar = slicer.qSlicerCLIProgressBar()
        self.cliProgressBar.visible = False 
        self.layout.addWidget(self.cliProgressBar)


        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()



        # -----------------------------------------------------------------------------------
        # DELETE THIS (default value)
        # -----------------------------------------------------------------------------------
        self.ui.notesFolderLineEdit_input.currentPath = "/home/luciacev"
        self.ui.notesFolderLineEdit_output.currentPath = "/home/luciacev"
        
        # Synchroniser les boutons radio avec la valeur du paramètre node
        self._syncModelTypeRadioWithParameterNode()
        self._syncNotesTypeRadioWithParameterNode()
        # Les boutons radio seront synchronisés avec le paramètre node après l'initialisation
    def _syncModelTypeRadioWithParameterNode(self):
        if not self._parameterNode:
            return
        if self._parameterNode.modelType == "Mini":
            self.ui.modelQuickRadioButton.checked = True
        elif self._parameterNode.modelType == "Max":
            self.ui.modelProRadioButton.checked = True
        else:
            self.ui.modelQuickRadioButton.checked = False
            self.ui.modelProRadioButton.checked = False

    def _syncNotesTypeRadioWithParameterNode(self):
        if not self._parameterNode:
            return
        if self._parameterNode.notesType == "TMJ":
            self.ui.notesTypeTMJRadioButton.checked = True
        elif self._parameterNode.notesType == "Ortho":
            self.ui.notesTypeOrthoRadioButton.checked = True
        else:
            self.ui.notesTypeTMJRadioButton.checked = False
            self.ui.notesTypeOrthoRadioButton.checked = False







    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode: CNE_UIParameterNode | None) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()
            # Synchroniser les boutons radio avec la valeur du paramètre node
            self._syncModelTypeRadioWithParameterNode()
            self._syncNotesTypeRadioWithParameterNode()



    def _checkCanApply(self, caller=None, event=None) -> None:
        """Active/désactive le bouton appliquer selon si les champs requis sont remplis"""
        if self._parameterNode:
            # Toujours activer le bouton
            self.ui.applyButton.toolTip = _("Renommer les fichiers")
            self.ui.applyButton.enabled = True


    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
                        
            print("\n" + "="*50)
            print("CNE (Clinical Notes Extraction)")
            print("="*50)           
            # Mettre à jour les paramètres depuis l'interface
            self._updateParameterNodeFromGUI()

            # Utiliser les valeurs du nœud de paramètres
            notesFolder_input = self._parameterNode.notesFolder_input
            modelType = self._parameterNode.modelType
            notesType = self._parameterNode.notesType
            notesFolder_output = self._parameterNode.notesFolder_output

            # --- English summary of selected parameters ---
            print(f"Input folder   : {notesFolder_input}")
            print(f"Output folder  : {notesFolder_output}")
            print(f"Selected model : {modelType}")
            print(f"Notes type     : {notesType}")
            print("-" * 50)

            # On récupère le cliNode renvoyé par la logic
            cliNode = self.logic.process(
                notesFolder_input, modelType,
                notesType, notesFolder_output
            )

            if cliNode:
                # On connecte le noeud à la barre de progression de l'UI
                self.cliProgressBar.setCommandLineModuleNode(cliNode)
                self.cliProgressBar.visible = True






    def _updateParameterNodeFromGUI(self) -> None:
        """Met à jour le nœud de paramètres à partir de l'interface utilisateur"""
        if not self._parameterNode:
            return
            
        wasModified = self._parameterNode.StartModify()
        
        # Get input/output folder paths from ctkPathLineEdit
        self._parameterNode.notesFolder_input = self.ui.notesFolderLineEdit_input.currentPath
        self._parameterNode.notesFolder_output = self.ui.notesFolderLineEdit_output.currentPath

        # Get selected model type radio button
        if self.ui.modelQuickRadioButton.checked:
            self._parameterNode.modelType = "Mini"
        elif self.ui.modelProRadioButton.checked:
            self._parameterNode.modelType = "Max"
        else:
            self._parameterNode.modelType = ""

        # Get selected notes type radio button
        if self.ui.notesTypeTMJRadioButton.checked:
            self._parameterNode.notesType = "TMJ"
        elif self.ui.notesTypeOrthoRadioButton.checked:
            self._parameterNode.notesType = "Ortho"
        else:
            self._parameterNode.notesType = ""

        self._parameterNode.EndModify(wasModified)




#
# CNE_UILogic
#
class CNE_UILogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return CNE_UIParameterNode(super().getParameterNode())

    def process(self, notesFolder_input: str, modelType: str, 
                notesType: str, notesFolder_output: str) -> bool:
        """
        Exécute le processus de summary de fichiers en utilisant le module CLI.
        """
        import os
        
        if not notesFolder_input or not modelType or not notesType or not notesFolder_output:
            logging.error("Process cancelled: Missing required parameters.")
            return None # Renvoie None en cas d'erreur
            
        os.makedirs(notesFolder_output, exist_ok=True)
            
        CLI_module = slicer.modules.cne_cli
        parameters = {
            "notesFolder_input": notesFolder_input,
            "modelType": modelType,
            "notesType": notesType,
            "notesFolder_output": notesFolder_output,
        }
        
        print("Launching background process (CLI)...")        
        self.cliNode = slicer.cli.run(CLI_module, None, parameters)
        
        # L'Observer pour les logs finaux reste dans la Logic, c'est très bien !
        self.cliNode.AddObserver(slicer.vtkMRMLCommandLineModuleNode.StatusModifiedEvent, self.onCliModified)
        
        # --- MODIFICATION ICI ---
        # On renvoie le noeud à l'interface au lieu de True
        return self.cliNode

    def onCliProgress(self, caller, event):
        """Callback déclenché à chaque <filter-progress> du CLI."""
        # GetProgress() renvoie un chiffre entre 0.0 et 1.0 (ou 0 et 100 selon la version)
        progress = caller.GetProgress() 
             
    def onCliModified(self, caller, event):
            """Callback triggered when CLI status changes (completed, cancelled, etc.)."""
            status = caller.GetStatus()
            
            if status & (slicer.vtkMRMLCommandLineModuleNode.Completed | slicer.vtkMRMLCommandLineModuleNode.Cancelled):
                print("Background process finished (CLI)")   
                print("="*50)
                
                # Handle success or cancellation
                if status == slicer.vtkMRMLCommandLineModuleNode.Completed:
                    print("CNE (Clinical Notes Extraction) SUCCESSFULL")
                    # Optional popup for the user
                    slicer.util.messageBox("Notes extraction is complete!")
                elif status == slicer.vtkMRMLCommandLineModuleNode.Cancelled:
                    print("PROCESS CANCELLED BY USER")
                    
                print("="*50)
            
                # Retrieve standard "prints" from the CLI
                output_text = caller.GetOutputText() 
                if output_text:
                    print("\n--- Detailed CLI Logs ---")
                    print(output_text.strip()) 
                    print("---------------------------\n")
                    
                # Retrieve Python errors from the CLI (Crucial for debugging!)
                error_text = caller.GetErrorText()
                if error_text:
                    logging.error("\n--- CLI ERRORS ---")
                    print(error_text.strip())
                    print("---------------------\n")

#
# CNE_UITest
#
class CNE_UITest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_CNE_UI1()

    def test_CNE_UI1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("CNE_UI1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = CNE_UILogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
