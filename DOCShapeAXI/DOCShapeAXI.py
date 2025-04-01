import os
import vtk, qt, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import subprocess
import platform
import sys
import time
import threading
import signal
import textwrap

from pathlib import Path
#
# DOCShapeAXI
#



class DOCShapeAXI(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "DOCShapeAXI" 
    self.parent.categories = ["Automated Dental Tools"]
    self.parent.dependencies = ["CondaSetUp"] 
    
    self.parent.contributors = ["Lucie Dole (University of North Carolina)", 
    "Gaelle Leroux (University of Michigan)",
    "Lucia Cevidanes (University of Michigan)",
    "Juan Carlos Prieto (University of North Carolina)"] 
    
    self.parent.helpText = textwrap.dedent("""
    This extension provides a Graphical User Interface (GUI) 
    for a deep learning automated classification of Alveolar Bone Defect in Cleft, 
    Nasopharynx Airway Obstruction and Mandibular Condyles.<br>

    - The input file must be a folder containing a list of vtk files.<br>

    - data type for classification: Mandibular Condyle, Nasopharynx Airway 
    Obstruction and Alveolar Bone Defect in Cleft.<br>

    - output directory: a folder that will contain all the outputs 
    (models, prediction and explainability results)<br>

    When prediction is over, you can open the output csv file which will containing 
    the path of each .vtk file as well as the predicted class. <br><br>

    More help can be found on the <a href="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools">Github repository</a> 
    for the extension.
    """).strip()

    self.parent.acknowledgementText = """
    This file was developed by Lucie Dole, (University of North Carolina) and 
    Gaelle Leroux (University of Michigan), and was supported by R01-DE024450.
    """


      
class DOCShapeAXIWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None) -> None:
    """Called when the user opens the module the first time and the widget is initialized."""
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation

    # self.logic = None
    self._parameterNode = None
    self._parameterNodeGuiTag = None
    self._updatingGUIFromParameterNode = False

    self.all_installed = False

    self.time_log = 0 # for progress bar
    self.progress = 0
    self.previous_time = 0
    self.start_time = 0

  def setup(self) -> None:
    self.removeObservers()

    """Called when the user opens the module the first time and the widget is initialized."""
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout. 
    uiWidget = slicer.util.loadUI(self.resourcePath("UI/DOCShapeAXI.ui"))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = DOCShapeAXILogic()

    # Connections
    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # UI elements 

    self.ui.browseDirectoryButton.connect('clicked(bool)',self.onBrowseOutputButton)
    self.ui.browseMountPointButton.connect('clicked(bool)',self.onBrowseMountPointButton)
    self.ui.cancelButton.connect('clicked(bool)', self.onCancel)

    self.ui.dataTypeComboBox.currentTextChanged.connect(self.onDataType)
    self.ui.resetButton.connect('clicked(bool)',self.onReset)

    self.ui.outputLineEdit.textChanged.connect(self.onEditOutputLine)
    self.ui.mountPointLineEdit.textChanged.connect(self.onEditMountPointLine)

    # initialize variables
    self.logic.output_dir = self.ui.outputLineEdit.text
    self.logic.input_dir = self.ui.mountPointLineEdit.text
    self.logic.data_type = self.ui.dataTypeComboBox.currentText

    self.ui.cancelButton.setHidden(True)
    self.ui.doneLabel.setHidden(True)

    
    self.time_log = 0
    self.progress = 0
    self.logic.cancel = False
    self.onCheckRequirements()

    self.ui.errorLabel.setVisible(False)
    self.ui.timeLabel.setVisible(False)
    self.ui.labelBar.setVisible(False)
    self.ui.labelBar.setStyleSheet(f"""QLabel{{font-size: 12px; qproperty-alignment: AlignCenter;}}""")
    self.ui.progressLabel.setVisible(False)
    self.ui.progressLabel.setStyleSheet(f"""QLabel{{font-size: 16px; qproperty-alignment: AlignCenter;}}""")
    self.ui.progressBar.setVisible(False)
    self.ui.progressBar.setRange(0,100)
    self.ui.progressBar.setTextVisible(True)

    self.ui.applyChangesButton.connect('clicked(bool)',self.onApplyChangesButton)

    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()

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
    # if self._parameterNode:
    #   self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
    #   self._parameterNodeGuiTag = None
    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

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

  def setParameterNode(self, inputParameterNode) -> None:
    """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

    if self._parameterNode:
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode:
      self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    # Initial GUI update
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return
    # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
    self._updatingGUIFromParameterNode = True

    # All the GUI updates are done
    self._updatingGUIFromParameterNode = False

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """
    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return
    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch
    self._parameterNode.EndModify(wasModified)


  ## 
  ## Inputs
  ##

  def onBrowseMountPointButton(self):
    mount_point = qt.QFileDialog.getExistingDirectory(self.parent, "Select a folder containing vtk files")
    if mount_point != '':
      self.logic.input_dir = mount_point
      self.ui.mountPointLineEdit.setText(self.logic.input_dir)

  def onEditMountPointLine(self):
    self.logic.input_dir = self.ui.mountPointLineEdit.text

  def onDataType(self):
    self.logic.data_type = self.ui.dataTypeComboBox.currentText
  
  ##
  ## Output
  ##
    
  def onBrowseOutputButton(self):
    newoutputFolder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a directory")
    if newoutputFolder != '':
      if newoutputFolder[-1] != "/":
        newoutputFolder += '/'
    self.outputFolder = newoutputFolder
    self.ui.outputLineEdit.setText(self.outputFolder)

  def onEditOutputLine(self):
    self.logic.output_dir = self.ui.outputLineEdit.text

  ##
  ##  Process
  ##
  def check_input_parameters(self): 
    msg = qt.QMessageBox()
    if not(os.path.isdir(self.logic.output_dir)):
      if not(os.path.isdir(self.logic.output_dir)):
        msg.setText("Output directory : \nIncorrect path.")
        print('Error: Incorrect path for output directory.')
        self.ui.outputLineEdit.setText('')
        print(f'output folder : {self.logic.output_dir}')
      else:
        msg.setText('Unknown error.')

      msg.setWindowTitle("Error")
      msg.exec_()
      return False

    elif not(os.path.isdir(self.logic.input_dir)):
      msg.setText("input file : \nIncorrect path.")
      print('Error: Incorrect path for input directory.')
      self.ui.mountPointLineEdit.setText('')

      msg.setWindowTitle("Error")
      msg.exec_()
      return False
    else:
      return True

  def onCheckRequirements(self):
    self.ui.labelBar.setVisible(False)
    self.ui.progressLabel.setVisible(False)

    if not self.logic.isCondaSetUp:
      self.ui.timeLabel.setText(f"Checking if SlicerConda is installed")
      messageBox = qt.QMessageBox()
      text = textwrap.dedent("""
      SlicerConda is not set up, please click 
      <a href=\"https://github.com/DCBIA-OrthoLab/SlicerConda/\">here</a> for installation.
      """).strip()
      messageBox.information(None, "Information", text)
      return False

    ## wsl
    if platform.system() == "Windows":

      self.ui.timeLabel.setHidden(False)
      self.ui.timeLabel.setText(f"Checking if wsl is installed, this task may take a moments")

      if self.logic.conda.testWslAvailable() : # if wsl is install
        self.ui.timeLabel.setText("WSL installed")
        if not self.logic.check_lib_wsl() : # if lib required are not install
          self.ui.timeLabel.setText(f"Checking if the required librairies are installed, this task may take a moments")
          messageBox = qt.QMessageBox()
          # text = "Code can't be launch. \nWSL doen't have all the necessary libraries, please download the installer and follow the instructin here : https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/wsl2_windows/installer_WSL2.zip may be blocked by Chrome, this is normal, just authorize it."
          text = textwrap.dedent("""
            WSL doesn't have all the necessary libraries, please download the installer 
            nd follow the instructions 
            <a href=\"https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/wsl2_windows/installer_WSL2.zip\">here</a> 
            for installation. The link may be blocked by Chrome, just authorize it.""").strip()

          messageBox.information(None, "Information", text)
          return False
      else : # if wsl not install, ask user to install it ans stop process
        messageBox = qt.QMessageBox()
        # text = "Code can't be launch. \nWSL is not installed, please download the installer and follow the instructin here : https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/wsl2_windows/installer_WSL2.zip may be blocked by Chrome, this is normal, just authorize it."
        text = textwrap.dedent("""
          WSL is not installed, please download the installer and follow the instructions 
          <a href=\"https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/wsl2_windows/installer_WSL2.zip\">here</a> 
          for installation. The link may be blocked by Chrome, just authorize it.""").strip()        

        messageBox.information(None, "Information", text)
        return False
    

    ## MiniConda

    self.ui.timeLabel.setText(f"Checking if miniconda is installed")
    if "Error" in self.logic.conda.condaRunCommand([self.logic.conda.getCondaExecutable(),"--version"]):
      messageBox = qt.QMessageBox()
      text = textwrap.dedent("""
      Code can't be launch. \nConda is not setup. 
      Please go the extension CondaSetUp in SlicerConda to do it.""").strip()
      messageBox.information(None, "Information", text)
      return False

    ## shapeAXI

    self.ui.timeLabel.setText(f"Checking if environnement exists")
    if not self.logic.conda.condaTestEnv(self.logic.name_env) : # check is environnement exist, if not ask user the permission to do it
      userResponse = slicer.util.confirmYesNoDisplay("The environnement to run the classification doesn't exist, do you want to create it ? ", windowTitle="Env doesn't exist")
      if userResponse :
        start_time = time.time()
        previous_time = start_time
        formatted_time = self.format_time(0)
        self.ui.timeLabel.setText(f"Creation of the new environment. This task may take a few minutes.\ntime: {formatted_time}")
        process = self.logic.install_shapeaxi()
        
        while self.logic.process.is_alive():
          slicer.app.processEvents()
          formatted_time = self.update_ui_time(start_time, previous_time)
          self.ui.timeLabel.setText(f"Creation of the new environment. This task may take a few minutes.\ntime: {formatted_time}")
    
        start_time = time.time()
        previous_time = start_time
        formatted_time = self.format_time(0)
        text = textwrap.dedent(f"""
        Installation of librairies into the new environnement. 
        This task may take a few minutes.\ntime: {formatted_time}""").strip()
        self.ui.timeLabel.setText(text)
      else:
        return False
    else:
      self.ui.timeLabel.setText(f"Ennvironnement already exists")
    
    ## pytorch3d

    self.ui.timeLabel.setText(f"Checking if pytorch3d is installed")
    if "Error" in self.logic.check_if_pytorch3d() : # pytorch3d not installed or badly installed 
      process = self.logic.install_pytorch3d()
      start_time = time.time()
      previous_time = start_time
    
      while self.logic.process.is_alive():
        slicer.app.processEvents()
        formatted_time = self.update_ui_time(start_time, previous_time)
        text = textwrap.dedent(f"""
        Installation of pytorch into the new environnement. 
        This task may take a few minutes.\ntime: {formatted_time}
        """).strip()
        self.ui.timeLabel.setText()
    else:
      self.ui.timeLabel.setText(f"pytorch3d is already installed")
      print("pytorch3d already installed")

    self.all_installed = True


  def update_ui_time(self, start_time, previous_time):
    current_time = time.time()
    gap=current_time-previous_time
    if gap>0.3:
      previous_time = current_time
      self.elapsed_time = current_time - start_time
      formatted_time = self.format_time(self.elapsed_time)
      return formatted_time

  def onApplyChangesButton(self):
    '''
    This function is called when the user want to run the prediction and explainabity script of ShapeAXI
    '''
    self.ui.applyChangesButton.setEnabled(False)
  
    if self.check_input_parameters() :

      self.ui.timeLabel.setHidden(False)
      self.ui.timeLabel.setText('time: 0.0s')
      slicer.app.processEvents()

      self.logic.check_cli_script()

      if 'Airway' in self.logic.data_type.split(' '):
        for task in ['binary', 'severity', 'regression']:
          if not self.logic.cancel :
            self.logic.task = task

            self.logic.setup_cli_command()
            self.onProcessStarted()

            start_time = time.time()
            previous_time = start_time

            while self.logic.process.is_alive():
              slicer.app.processEvents()
              self.onProcessUpdate()
              formatted_time = self.update_ui_time(start_time, previous_time)
              self.ui.timeLabel.setText(f"time : {formatted_time}")

            self.resetProgressBar()
        self.onProcessCompleted()
      else:
        self.logic.task = 'severity'
        self.logic.setup_cli_command()
        self.onProcessStarted()

        start_time = time.time()
        previous_time = start_time
        while self.logic.process.is_alive():
          slicer.app.processEvents()
          self.onProcessUpdate()
          formatted_time = self.update_ui_time(start_time, previous_time)
          self.ui.timeLabel.setText(f"time : {formatted_time}")
        self.onProcessCompleted()

      self.ui.applyChangesButton.setEnabled(True)
      self.ui.cancelButton.setHidden(True)
  
  def resetProgressBar(self):
    self.ui.progressBar.setValue(0)
    self.logic.progress = 0
    self.logic.previous_saxi_task='predict'

    self.ui.timeLabel.setVisible(False)
    self.ui.labelBar.setVisible(False)
    self.ui.progressLabel.setText('Prediction in progress...')
    self.ui.progressLabel.setVisible(False)
    
    self.ui.progressBar.setVisible(False)
    self.ui.progressBar.setRange(0,100)
    self.ui.progressBar.setTextVisible(True)
    self.ui.progressBar.setValue(0)
    self.ui.progressBar.setFormat("")

  def onProcessStarted(self):
    self.logic.count_num_subjects()
    self.ui.progressBar.setValue(0)
    self.ui.labelBar.setText(f'Loading {self.logic.task} model...')
    
    self.ui.applyChangesButton.setEnabled(False)
    self.ui.doneLabel.setHidden(True)
    self.ui.cancelButton.setHidden(False)
    self.ui.labelBar.setHidden(False)
    self.ui.timeLabel.setHidden(False)
    self.ui.progressLabel.setHidden(False)
    self.ui.progressBar.setHidden(False)
    formatted_time = self.format_time(0)
    self.ui.timeLabel.setText(f"time : {formatted_time}")

  def onProcessUpdate(self):
    if os.path.isfile(self.logic.log_path):
      time_progress = os.path.getmtime(self.logic.log_path)
      line = self.logic.read_log_path()
      if (time_progress != self.time_log) and line:
        self.task, current_saxi_task, progress, num_classes = line.strip().split(',')

        if progress == 'NaN':
          self.ui.labelBar.setText(f'Loading {self.task} model...')
        else:
          self.progress = int(progress)
          if self.logic.previous_saxi_task != current_saxi_task: 
            self.progress = 0
            self.logic.previous_saxi_task = current_saxi_task
            self.ui.progressBar.setValue(0)

          self.ui.progressLabel.setText(f'{current_saxi_task} in progress...')
          self.ui.labelBar.setText(f"{self.task} model \nNumber of processed subjects : {self.progress}/{self.logic.nbSubjects}")
          progressbar_value = round((self.progress) /self.logic.nbSubjects * 100,2)
          self.time_log = time_progress

          self.ui.progressBar.setValue(progressbar_value)
          self.ui.progressBar.setFormat(str(progressbar_value)+"%")

  def onProcessCompleted(self):

    if self.logic.cancel:
      self.ui.doneLabel.setText("Process cancelled by user")
      self.ui.doneLabel.setStyleSheet(f"""QLabel{{font-size: 20px; qproperty-alignment: AlignCenter; color:'red';}}""")

    elif self.logic.stderr == '':
      self.ui.doneLabel.setText("Process completed successfully")
      self.ui.doneLabel.setStyleSheet(f"""QLabel{{font-size: 20px; qproperty-alignment: AlignCenter; color:'green';}}""")

    else:
      self.ui.doneLabel.setText("An error has occured.\nSee below the error message.")
      self.ui.doneLabel.setStyleSheet(f"""QLabel{{font-size: 20px; qproperty-alignment: AlignCenter; color:'red';}}""")
      self.ui.errorLabel.setText(self.logic.stderr)
      self.ui.errorLabel.setVisible(True)

    self.ui.applyChangesButton.setEnabled(True)
    self.ui.resetButton.setEnabled(True)
    self.ui.progressLabel.setHidden(False)     
    self.ui.cancelButton.setHidden(True)
    self.resetProgressBar()
    self.ui.doneLabel.setHidden(False)
    
    formatted_time = self.format_time(self.elapsed_time)
    self.ui.timeLabel.setText(f"time : {formatted_time}")
    self.ui.timeLabel.setHidden(False)

    self.ui.doneLabel.setHidden(False)
      
  def onReset(self):
    self.logic.cancel = False
    self.ui.outputLineEdit.setText("")
    self.ui.mountPointLineEdit.setText("")
    self.ui.errorLabel.setText("Error: ")
    self.ui.errorLabel.setVisible(False)

    self.ui.applyChangesButton.setEnabled(True)
    self.resetProgressBar()
    self.ui.progressLabel.setHidden(True)
    self.ui.doneLabel.setHidden(True)
    self.ui.timeLabel.setHidden(True)
    formatted_time = self.format_time(0)
    self.ui.timeLabel.setText(f"time : {formatted_time}")
    self.ui.progressBar.setEnabled(False)
    self.ui.progressBar.setRange(0,100)
    self.removeObservers()  
    self.ui.cancelButton.setEnabled(True)

  def onCancel(self):
    self.ui.labelBar.setText(f'Cancelling process...')

    self.logic.cancel_process()

    self.ui.cancelButton.setEnabled(False)
    self.removeObservers()  
    self.onReset()

  def format_time(self,seconds):
    """ Convert seconds to H:M:S format. """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


class DOCShapeAXILogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py

  """

  def __init__(self):
    
    """Called when the logic class is instantiated. Can be used for initializing member variables."""
    ScriptedLoadableModuleLogic.__init__(self)

    self.isCondaSetUp = False
    self.isWsl = False
    self.name_env = 'shapeaxi'
  
    self.cancel = False
    self.progress = 0
    self.previous_saxi_task='predict'

    self.check_log_path()
    self.conda = self.init_conda()

  def check_log_path(self):
    self.log_path = os.path.normpath(os.path.join(os.path.dirname(__file__), 'process.log'))
    
    if '\\' in self.log_path:
      self.log_path = self.log_path.replace('\\', '/')
    
    with open(self.log_path, mode='w') as f: pass

  def init_conda(self):
    # check if CondaSetUp exists
    try:
      import CondaSetUp
    except:
      return False
    self.isCondaSetUp = True

    # set up conda on windows with WSL
    if platform.system() == "Windows":
      from CondaSetUp import CondaSetUpCallWsl
      return CondaSetUpCallWsl()
    else:
      from CondaSetUp import CondaSetUpCall
      return CondaSetUpCall()     

  def run_conda_command(self, target, command):
    self.process = threading.Thread(target=target, args=command) #run in parallel to not block slicer
    self.process.start()

  def install_shapeaxi(self):
    self.run_conda_command(target=self.conda.condaCreateEnv, command=(self.name_env,"3.9",["shapeaxi==1.0.10"],)) #run in parallel to not block slicer

  def check_if_pytorch3d(self):
    conda_exe = self.conda.getCondaExecutable()
    command = [conda_exe, "run", "-n", self.name_env, "python" ,"-c", f"\"import pytorch3d;import pytorch3d.renderer\""]
    return self.conda.condaRunCommand(command)
    
  def install_pytorch3d(self):
    result_pythonpath = self.check_pythonpath_windows("DOCShapeAXI_utils.install_pytorch")
    if not result_pythonpath :
      self.give_pythonpath_windows()
      result_pythonpath = self.check_pythonpath_windows("DOCShapeAXI_utils.install_pytorch")
      
    if result_pythonpath : 
      conda_exe = self.conda.getCondaExecutable()
      path_pip = self.conda.getCondaPath()+f"/envs/{self.name_env}/bin/pip"
      command = [conda_exe, "run", "-n", self.name_env, "python" ,"-m", f"DOCShapeAXI_utils.install_pytorch",path_pip]

    self.run_conda_command(target=self.conda.condaRunCommand, command=(command,))

  def setup_cli_command(self):

    args = self.find_cli_parameters()
    conda_exe = self.conda.getCondaExecutable()
    command = [conda_exe, "run", "-n", self.name_env, "python" ,"-m", f"DOCShapeAXI_CLI"]
    for arg in args :
      command.append("\""+arg+"\"")

    self.run_conda_command(target=self.condaRunCommand, command=(command,))

  def check_lib_wsl(self)->bool:
    '''
    Check if wsl contains the require librairies
    '''
    result1 = subprocess.run("wsl -- bash -c \"dpkg -l | grep libxrender1\"", capture_output=True, text=True)
    output1 = result1.stdout.encode('utf-16-le').decode('utf-8')
    clean_output1 = output1.replace('\x00', '')

    result2 = subprocess.run("wsl -- bash -c \"dpkg -l | grep libgl1-mesa-glx\"", capture_output=True, text=True)
    output2 = result2.stdout.encode('utf-16-le').decode('utf-8')
    clean_output2 = output2.replace('\x00', '')

    return "libxrender1" in clean_output1 and "libgl1-mesa-glx" in clean_output2

  def check_pythonpath_windows(self,file):
      '''
      Check if the environment env_name in wsl know the path to a specific file (ex : Crownsegmentationcli.py)
      return : bool
      '''
      conda_exe = self.conda.getCondaExecutable()
      command = [conda_exe, "run", "-n", self.name_env, "python" ,"-c", f"\"import {file} as check;import os; print(os.path.isfile(check.__file__))\""]
      result = self.conda.condaRunCommand(command)
      print("output CHECK python path: ", result)
      if "True" in result :
          return True
      return False

  def give_pythonpath_windows(self):
      '''
      take the pythonpath of Slicer and give it to the environment name_env in wsl.
      '''
      paths = slicer.app.moduleManager().factoryManager().searchPaths
      mnt_paths = []
      for path in paths :
          mnt_paths.append(f"\"{self.windows_to_linux_path(path)}\"")
      pythonpath_arg = 'PYTHONPATH=' + ':'.join(mnt_paths)
      conda_exe = self.conda.getCondaExecutable()
      argument = [conda_exe, 'env', 'config', 'vars', 'set', '-n', self.name_env, pythonpath_arg]
      results = self.conda.condaRunCommand(argument)
      print("output GIVE python path: ", results)

  def windows_to_linux_path(self,windows_path):
    '''
    Convert a windows path to a wsl path
    '''
    windows_path = windows_path.strip()

    path = windows_path.replace('\\', '/')

    if ':' in path:
      drive, path_without_drive = path.split(':', 1)
      path = "/mnt/" + drive.lower() + path_without_drive

    return path
  
  def check_cli_script(self):
    if not self.check_pythonpath_windows("DOCShapeAXI_CLI") : 
      self.give_pythonpath_windows()
      results = self.check_pythonpath_windows("DOCShapeAXI_CLI")

  
  def condaRunCommand(self, command: list[str]):
    '''
    Runs a command in a specified Conda environment, handling different operating systems.
    
    copy paste from SlicerConda and change the process line to be able to get the stderr/stdout 
    and cancel the process without blocking slicer
    '''
    path_activate = self.conda.getActivateExecutable()

    if path_activate=="None":
        return "Path to conda no setup"

    if platform.system() == "Windows":

      command_execute = f"source {path_activate} {self.name_env} &&"
      for com in command :
        command_execute = command_execute+ " "+com

      user = self.conda.getUser()
      command_to_execute = ["wsl", "--user", user,"--","bash","-c", command_execute]
      print("command_to_execute in condaRunCommand : ",command_to_execute)

      self.subpro = subprocess.Popen(command_to_execute, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              text=True, encoding='utf-8', errors='replace', env=slicer.util.startupEnvironment(),
                              creationflags=subprocess.CREATE_NEW_PROCESS_GROUP  # For Windows
                              )
    else:
        
      path_conda_exe = self.conda.getCondaExecutable()
      command_execute = f"{path_conda_exe} run -n {self.name_env}"
      for com in command :
          command_execute = command_execute+ " "+com

      print("command_execute dans conda run : ",command_execute)
      self.subpro = subprocess.Popen(command_execute, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', env=slicer.util.startupEnvironment(), executable="/bin/bash", preexec_fn=os.setsid)
  
    self.stdout, self.stderr = self.subpro.communicate()

  def cancel_process(self):
    if platform.system() == 'Windows':
      self.subpro.send_signal(signal.CTRL_BREAK_EVENT)
    else:
      os.killpg(os.getpgid(self.subpro.pid), signal.SIGTERM)
    print("Cancellation requested. Terminating process...")

    self.subpro.wait() ## important
    self.cancel = True

  def read_log_path(self):
    with open(self.log_path, 'r') as f:
      line = f.readline()
      if line != '':
        return line

  def count_num_subjects(self):
    self.nbSubjects = 0
    for elt in os.listdir(self.input_dir):
      name, ext = os.path.splitext(elt)
      if ext == '.vtk':
        self.nbSubjects +=1

  def find_cli_parameters(self):

    self.nn_type = self.find_nn_type()
    self.model, self.num_classes = self.find_model_name()

    parameters = [self.input_dir, 
                  self.output_dir, 
                  self.data_type, 
                  self.task, 
                  self.model, 
                  self.nn_type, 
                  str(self.num_classes), 
                  self.log_path]
    
    return parameters
        
  def find_nn_type(self):
    if self.task == 'regression':
      return 'SaxiMHAFBRegression'
    else:
      return 'SaxiMHAFBClassification'

  def find_model_name(self):
    if 'Condyle' in self.data_type.split(' '):
      model_name='condyles_4_class'
      self.num_classes = 4

    elif 'Airway' in self.data_type.split(' '):
      if self.task == 'binary':
        model_name='airways_2_class'
        self.num_classes = 2

      elif self.task == 'severity':
        model_name='airways_4_class'
        self.num_classes = 4

      elif self.task == 'regression':
        model_name='airways_4_regress'
        self.num_classes = 1
      else:
        print("no model found for undefined task")

    elif 'Cleft' in self.data_type.split(' '):
      model_name='clefts_4_class'
      self.num_classes = 4

    else:
      print("No model found")
      return None, None
    return model_name, self.num_classes