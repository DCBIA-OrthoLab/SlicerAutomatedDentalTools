import vtk
import numpy as np
import slicer
import logging
import os
import sys
from pathlib import Path
from enum import Flag, auto
import qt
import sys

# ===== Logging Configuration =====
logger = logging.getLogger("VFACE_segmentation_logic")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Desactivate les warnings VTK
vtk.vtkObject.GlobalWarningDisplayOff()

# ─── Model descriptions ──────────────────────────────────────────────────────

MODEL_DESCRIPTIONS = {
    "DentalSegmentator": (
        "DentalSegmentator - Segments: Upper Skull (includes Maxilla), Mandible, Mandibular Canal, Upper Teeth, Lower Teeth - Designed for permanent dentition."
    ),
    "PediatricDentalsegmentator": (
        "PediatricDentalsegmentator - Segments: Upper Skull (includes Maxilla), Mandible, Mandibular Canal, Upper Teeth, Lower Teeth - Designed for mixed dentition (baby and permanent teeth)."
    ),
    "NasoMaxillaDentSeg": (
        "NasoMaxillaDentSeg - Segments: Upper Skull, separate Maxilla, Mandible, Mandibular Canal, Upper Teeth, Lower Teeth - Designed for permanent dentition."
    ),
    "UniversalLabDentalsegmentator": (
        "UniversalLabDentalsegmentator - Segments: Upper Skull, Mandibular Canal, All teeth - Designed for mixed and Permanent dentition."
    ),
}

# ─── Export formats enumeration ───────────────────────────────────────────────

class ExportFormat(Flag):
    OBJ = auto()
    STL = auto()
    NIFTI = auto()
    GLTF = auto()
    VTK = auto()
    VTK_MERGED = auto()

class PythonDependencyChecker:
    """Check python dependency"""
    
    def downloadWeightsIfNeeded(self, onLine):
        """Check and download the weights if necessary"""
        onLine("Weights check completed")
        return True

# ─── Segmentation Logic ───────────────────────────────────────────

class SegmentationLogic:
    """
    Class containing all the dental segmentation logic without UI
    """
    
    def __init__(self):
        self.folderPath = ""
        self.folderFiles = []
        self.currentFileIndex = 0
        self.currentVolumeNode = None
        self.outputFolderPath = ""
        self.processedVolumes = {}
        self.isStopping = False
        self._minimumIslandSize_mm3 = 60
        self.logic = self._createSlicerSegmentationLogic()
        self._dependencyChecker = PythonDependencyChecker()
        self.fullInfoLogs = []
        
        #Default Configuration
        self.selectedModel = "DentalSegmentator"
        self.selectedDevice = "cuda"
        self.exportFormats = ExportFormat.STL | ExportFormat.NIFTI
    
    def setInputFolder(self, folderPath):
        """Define input folder"""
        self.folderPath = folderPath
        folder = Path(folderPath)
        # Filtrer selon vos formats, ex. tous les fichiers NIfTI
        self.folderFiles = list(folder.rglob("*.nii*")) + list(folder.rglob("*.gipl")) + list(folder.rglob("*.gipl.gz"))
        self.currentFileIndex = 0
        self.log_info(f"Found {len(self.folderFiles)} file(s) in the folder.")
    
    def setOutputFolder(self, outputPath):
        """Define output folder"""
        self.outputFolderPath = outputPath
    
    def setModel(self, model_name):
        """Define model to use"""
        if model_name in MODEL_DESCRIPTIONS:
            self.selectedModel = model_name
            self.log_info(f"Model set to: {model_name}")
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def setDevice(self, device):
        """Define the device (cuda, cpu, mps)"""
        self.selectedDevice = device
    
    def setExportFormats(self, formats):
        """Define export formats"""
        self.exportFormats = formats
    
    def log_info(self, message):
        """Log d'information"""
        logger.info(f"[SEGMENTATION] {message}")
        self.fullInfoLogs.append(message)
    
    def log_error(self, message):
        """Error log"""
        logger.error(f"[ERROR] {message}")
        self.fullInfoLogs.append(f"ERROR: {message}")
    
    def processAllFiles(self):
        """Process all input files"""
        if not self.folderPath or not self.folderFiles:
            self.log_error("No input folder or files specified")
            return False
        
        if not self.outputFolderPath:
            self.log_error("No output folder specified")
            return False
        
        # Install the dependencies
        if not self._installDependencies():
            return False
        
        # Processing all files
        for i, file_path in enumerate(self.folderFiles):
            self.currentFileIndex = i
            self.log_info(f"Processing file {i+1}/{len(self.folderFiles)}: {file_path.name}")
            
            # Keep Slicer dynamic
            slicer.app.processEvents()
            
            if self.isStopping:
                self.log_info("Processing stopped by user")
                break
            
            try:
                success = self.processFile(file_path)
                if not success:
                    self.log_error(f"Failed to process file: {file_path}")
                    continue
            except Exception as e:
                self.log_error(f"Exception processing file {file_path}: {str(e)}")
                continue
            
            slicer.app.processEvents()
        
        self.log_info("All files processing completed")
        return True
    
    def processFile(self, file_path):
        """Traite un seul fichier"""
        try:
            #Load volume
            loadedVolume = slicer.util.loadVolume(str(file_path))
            if not loadedVolume:
                self.log_error(f"Failed to load volume: {file_path}")
                return False
            
            self.currentVolumeNode = loadedVolume
            self.log_info(f"Loaded volume: {loadedVolume.GetName()}")
            
            # Configuration of display
            slicer.util.setSliceViewerLayers(background=loadedVolume)
            slicer.util.resetSliceViews()
            
            # Run segmentation
            success = self._runSegmentationForVolume(loadedVolume)
            
            return success
            
        except Exception as e:
            self.log_error(f"Error in processFile: {str(e)}")
            return False
    
    def _installDependencies(self):
        """Install the dependencies"""
        try:
            self.log_info("Checking dependencies...")
            
            if not self.isNNUNetModuleInstalled():
                self.log_error("NNUNet module not installed")
                return False
            
            if not self._installNNUNetIfNeeded():
                return False
            
            if not self._dependencyChecker.downloadWeightsIfNeeded(self.log_info):
                return False
            
            self.log_info("Dependencies check completed")
            return True
            
        except Exception as e:
            self.log_error(f"Error installing dependencies: {str(e)}")
            return False
    
    def _runSegmentationForVolume(self, volumeNode):
        """Run the segmentation"""
        try:
            from SlicerNNUNetLib import Parameter
            
            #Model Configuration
            parameter = self._getModelParameter()
            
            if not parameter.isSelectedDeviceAvailable():
                self.log_info(f"Selected device ({parameter.device.upper()}) not available, falling back to CPU")
            
            slicer.app.processEvents()
            self.logic.setParameter(parameter)
            
            #Start the segmentation
            self.logic.startSegmentation(volumeNode)
            
            # Wait end of segmentation
            self._waitForSegmentationWithEvents()
            
            # Process results
            return self._processSegmentationResults(volumeNode)
            
        except Exception as e:
            self.log_error(f"Error in segmentation: {str(e)}")
            return False
    
    def _waitForSegmentationWithEvents(self):
        """Wait the end of the segmentation"""
        import time
        
        start_time = time.time()
        last_log_time = start_time
        timeout_seconds = 3600  # 1 hour maximum
        
        self.log_info("Segmentation started - this may take several minutes...")
        
        while not self.isStopping:
            slicer.app.processEvents()
            
            # Check if the segmentation is over
            segmentation_finished = False
            try:
                # Try different version depending on the module
                if hasattr(self.logic, 'isFinished'):
                    segmentation_finished = self.logic.isFinished()
                elif hasattr(self.logic, 'finished'):
                    segmentation_finished = self.logic.finished
                elif hasattr(self.logic, 'isRunning'):
                    segmentation_finished = not self.logic.isRunning()
                elif hasattr(self.logic, 'running'):
                    segmentation_finished = not self.logic.running
                else:
                    try:
                        test_seg = self.logic.loadSegmentation()
                        if test_seg:
                            segmentation_finished = True
                    except:
                        segmentation_finished = False
                        
            except Exception as e:
                self.log_error(f"Error checking segmentation status: {str(e)}")
                # If error just wait
                segmentation_finished = False
            
            if segmentation_finished:
                break
            
            time.sleep(0.05)
            
            current_time = time.time()
            
            #progression Log
            if current_time - last_log_time > 30:
                elapsed = int(current_time - start_time)
                minutes = elapsed // 60
                seconds = elapsed % 60
                self.log_info(f"Segmentation in progress... ({minutes}m {seconds}s)")
                last_log_time = current_time
            
            # Check timeout
            if current_time - start_time > timeout_seconds:
                self.log_error("Segmentation timeout - process taking too long")
                self.logic.stopSegmentation()
                return False
            
            if self.isStopping:
                break
        
        if self.isStopping:
            self.log_info("Segmentation stopped by user")
            return False
        
        self.log_info("Segmentation completed")
        return True
    
    def _getModelParameter(self):
        """Get Model parameters"""
        from SlicerNNUNetLib import Parameter
        
        if self.selectedModel == "PediatricDentalsegmentator":
            basePath = Path(__file__).parent.joinpath("Resources", "ML", "Dataset001_380CT", "nnUNetTrainer__nnUNetPlans__3d_fullres").resolve()
            self._downloadModelIfNeeded("pediatricdentalseg", basePath)
            
        elif self.selectedModel == "NasoMaxillaDentSeg":
            basePath = Path(__file__).parent.joinpath("Resources", "ML", "Dataset001_max4", "nnUNetTrainer__nnUNetPlans__3d_fullres").resolve()
            self._downloadModelIfNeeded("nasomaxilladentseg", basePath)
            
        elif self.selectedModel == "UniversalLabDentalsegmentator":
            basePath = Path(__file__).parent.joinpath("Resources", "ML", "Dataset002_380CT", "nnUNetTrainer__nnUNetPlans__3d_fullres").resolve()
            self._downloadModelIfNeeded("universallab", basePath)
            
        else:  # Default DentalSegmentator
            self.log_info("Using Dataset111_453CT for DentalSegmentator")
            basePath = Path(__file__).parent.parent.joinpath("Resources", "ML", "Dataset111_453CT", "nnUNetTrainer__nnUNetPlans__3d_fullres").resolve()
        
        return Parameter(folds="0", modelPath=basePath, device=self.selectedDevice)
    
    def _downloadModelIfNeeded(self, model_type, basePath):
        """Download the model"""
        fold_path = basePath.joinpath("fold_0")
        fold_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = fold_path.joinpath("checkpoint_final.pth")
        
        if not checkpoint.exists():
            self.log_info(f"Downloading {model_type} model...")
            
            urls = {
                "pediatricdentalseg": {
                    "checkpoint": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/PEDIATRICDENTALSEG_MODEL/checkpoint_final.pth",
                    "dataset": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/PEDIATRICDENTALSEG_MODEL/dataset.json",
                    "plans": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/PEDIATRICDENTALSEG_MODEL/plans.json"
                },
                "nasomaxilladentseg": {
                    "checkpoint": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/NASOMAXILLADENTSEG_MODEL/checkpoint_final.pth",
                    "dataset": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/NASOMAXILLADENTSEG_MODEL/dataset.json",
                    "plans": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/NASOMAXILLADENTSEG_MODEL/plans.json"
                },
                "universallab": {
                    "checkpoint": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/UNIVERSALLAB_MODEL/checkpoint_final.pth",
                    "dataset": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/UNIVERSALLAB_MODEL/dataset.json",
                    "plans": "https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/UNIVERSALLAB_MODEL/plans.json"
                }
            }
            
            model_urls = urls.get(model_type)
            if model_urls:
                slicer.util.downloadFile(model_urls["checkpoint"], str(checkpoint))
                slicer.util.downloadFile(model_urls["dataset"], str(basePath.joinpath("dataset.json")))
                slicer.util.downloadFile(model_urls["plans"], str(basePath.joinpath("plans.json")))
    
    def _processSegmentationResults(self, volumeNode):
        """Process segmentation results"""
        try:
            #Load results
            segmentationNode = self._loadSegmentationResults()
            if not segmentationNode:
                self.log_error("No segmentation results found")
                return False
            
            segmentationNode.SetName(volumeNode.GetName() + "_Segmentation")
            
            # Display progress
            self._updateSegmentationDisplay(segmentationNode)
            
            slicer.app.processEvents()
            
            # Export selected formats
            if self.exportFormats & ExportFormat.NIFTI:
                self.log_info("Starting NIfTI export...")
                self._saveSegmentationAsNifti(segmentationNode, volumeNode)
                slicer.app.processEvents()
            
            if self.exportFormats & ExportFormat.STL:
                self.log_info("Starting STL export...")
                self._exportSTL(segmentationNode)
                slicer.app.processEvents()
            
            if self.exportFormats & ExportFormat.OBJ:
                self.log_info("Starting OBJ export...")
                self._exportOBJ(segmentationNode)
                slicer.app.processEvents()
            
            if self.exportFormats & ExportFormat.VTK_MERGED:
                self.log_info("Starting merged VTK export...")
                self._exportMergedVTK(segmentationNode)
                slicer.app.processEvents()
            
            if self.exportFormats & ExportFormat.VTK:
                self.log_info("Starting per-label VTK export...")
                self._exportVTKPerLabel(segmentationNode)
                slicer.app.processEvents()
            
            # Cleaning
            self._cleanupAfterCase(volumeNode, segmentationNode)
            
            return True
            
        except Exception as e:
            self.log_error(f"Error processing results: {str(e)}")
            return False
    
    def _loadSegmentationResults(self):
        """Load segmentation results"""
        try:
            segmentationNode = self.logic.loadSegmentation()
            return segmentationNode
        except Exception as e:
            self.log_error(f"Error loading segmentation: {str(e)}")
            return None
    
    def _updateSegmentationDisplay(self, segmentationNode):
        """Update display of the segmentation"""
        if not segmentationNode:
            return
        
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(self.currentVolumeNode)
        
        if not segmentationNode.GetDisplayNode():
            segmentationNode.CreateDefaultDisplayNodes()
            slicer.app.processEvents()
        
        segmentation = segmentationNode.GetSegmentation()
        
        # Apply colors and labels to the segmentation
        self._applySegmentationLabelsAndColors(segmentation)
    
    def _applySegmentationLabelsAndColors(self, segmentation):
        """Apply colors and labels to the segmentation"""
        if self.selectedModel == "UniversalLabDentalsegmentator":
            labels = [
                "Upper-right third molar", "Upper-right second molar", "Upper-right first molar",
                "Upper-right second premolar", "Upper-right first premolar", "Upper-right canine",
                "Upper-right lateral incisor", "Upper-right central incisor", "Upper-left central incisor",
                "Upper-left lateral incisor", "Upper-left canine", "Upper-left first premolar",
                "Upper-left second premolar", "Upper-left first molar", "Upper-left second molar",
                "Upper-left third molar", "Lower-left third molar", "Lower-left second molar",
                "Lower-left first molar", "Lower-left second premolar", "Lower-left first premolar",
                "Lower-left canine", "Lower-left lateral incisor", "Lower-left central incisor",
                "Lower-right central incisor", "Lower-right lateral incisor", "Lower-right canine",
                "Lower-right first premolar", "Lower-right second premolar", "Lower-right first molar",
                "Lower-right second molar", "Lower-right third molar", "Upper-right second molar (baby)",
                "Upper-right first molar (baby)", "Upper-right canine (baby)",
                "Upper-right lateral incisor (baby)", "Upper-right central incisor (baby)",
                "Upper-left central incisor (baby)", "Upper-left lateral incisor (baby)",
                "Upper-left canine (baby)", "Upper-left first molar (baby)",
                "Upper-left second molar (baby)", "Lower-left second molar (baby)",
                "Lower-left first molar (baby)", "Lower-left canine (baby)",
                "Lower-left lateral incisor (baby)", "Lower-left central incisor (baby)",
                "Lower-right central incisor (baby)", "Lower-right lateral incisor (baby)",
                "Lower-right canine (baby)", "Lower-right first molar (baby)",
                "Lower-right second molar (baby)", "Mandible", "Maxilla", "Mandibular canal"
            ]
        elif self.selectedModel == "NasoMaxillaDentSeg":
            labels = ["Upper Skull", "Mandible", "Upper Teeth", "Lower Teeth", "Mandibular canal", "Maxilla"]
        else:
            labels = ["Upper Skull", "Mandible", "Upper Teeth", "Lower Teeth", "Mandibular canal"]
        
        # Application des labels
        segmentIds = list(segmentation.GetSegmentIDs())
        for i, (segmentId, label) in enumerate(zip(segmentIds, labels)):
            segment = segmentation.GetSegment(segmentId)
            if segment:
                segment.SetName(label)
    
    def _saveSegmentationAsNifti(self, segmentationNode, volumeNode):
        """Sauvegarde la segmentation au format NIfTI"""
        try:
            self.log_info("Saving segmentation as NIfTI")
            
            if volumeNode:
                segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
            
            labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            success = slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
                segmentationNode, labelmapVolumeNode, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY)
            
            if not success:
                self.log_error("Failed to export segments to labelmap")
                return False
            
            output_path = os.path.join(self.outputFolderPath, segmentationNode.GetName() + ".nii.gz")
            saved = slicer.util.saveNode(labelmapVolumeNode, output_path)
            
            if saved:
                self.log_info(f"Segmentation saved to {output_path}")
            else:
                self.log_error(f"Failed to save segmentation to {output_path}")
            
            # Nettoyage du label-map temporaire
            slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
            return saved
            
        except Exception as e:
            self.log_error(f"Error saving NIfTI: {str(e)}")
            return False
    
    def _exportSTL(self, segmentationNode):
        """Export au format STL"""
        try:
            self.log_info("Exporting to STL format")
            slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsClosedSurfaceRepresentationToFiles(
                self.outputFolderPath, segmentationNode, None, "STL", True, 1.0, False
            )
        except Exception as e:
            self.log_error(f"Error exporting STL: {str(e)}")
    
    def _exportOBJ(self, segmentationNode):
        """Export au format OBJ"""
        try:
            self.log_info("Exporting to OBJ format")
            slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsClosedSurfaceRepresentationToFiles(
                self.outputFolderPath, segmentationNode, None, "OBJ", True, 1.0, False
            )
        except Exception as e:
            self.log_error(f"Error exporting OBJ: {str(e)}")
    
    def _exportMergedVTK(self, segmentationNode):
        """Export VTK"""
        try:
            import os, re
            from vtk.util.numpy_support import vtk_to_numpy
            
            self.log_info("MergedVTK: Start")
            
            # Create labelmap
            labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segmentationNode, labelmapVolumeNode)
            img = labelmapVolumeNode.GetImageData()

            # Marching Cubes
            self.log_info("MergedVTK: MarchingCubes")
            mc = vtk.vtkDiscreteMarchingCubes()
            mc.SetInputData(img)
            for l in np.unique(vtk_to_numpy(img.GetPointData().GetScalars())):
                if l: 
                    mc.SetValue(int(l), int(l))
            mc.Update()

            # Clean + smooth
            self.log_info("MergedVTK: Cleaning + smoothing")
            clean = vtk.vtkCleanPolyData()
            clean.SetInputConnection(mc.GetOutputPort())
            clean.Update()
            
            ws = vtk.vtkWindowedSincPolyDataFilter()
            ws.SetInputConnection(clean.GetOutputPort())
            ws.SetNumberOfIterations(60)
            ws.SetPassBand(0.05)
            ws.BoundarySmoothingOn()
            ws.FeatureEdgeSmoothingOn()
            ws.NonManifoldSmoothingOn()
            ws.NormalizeCoordinatesOn()
            ws.Update()

            # Normales
            self.log_info("MergedVTK: Computing normals")
            flatN = vtk.vtkPolyDataNormals()
            flatN.SetInputConnection(ws.GetOutputPort())
            flatN.ComputePointNormalsOff()
            flatN.ComputeCellNormalsOn()
            flatN.SplittingOff()
            flatN.AutoOrientNormalsOn()
            flatN.ConsistencyOn()
            flatN.SetFeatureAngle(180)
            flatN.Update()

            rawPoly = flatN.GetOutput()
            labelArray = rawPoly.GetCellData().GetScalars()
            labels = np.unique(vtk_to_numpy(labelArray))
            append = vtk.vtkAppendPolyData()

            # Parcours des labels
            for i, labelValue in enumerate(labels, start=1):
                if labelValue == 0:
                    continue
                self.log_info(f"MergedVTK: Processing label {int(labelValue)} ({i}/{len(labels)})")
                
                slicer.app.processEvents()

                thresh = vtk.vtkThreshold()
                thresh.SetInputData(rawPoly)
                thresh.SetInputArrayToProcess(0, 0, 0,
                    vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,
                    labelArray.GetName())
                thresh.SetLowerThreshold(labelValue)
                thresh.SetUpperThreshold(labelValue)
                thresh.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
                thresh.Update()

                surf = vtk.vtkDataSetSurfaceFilter()
                surf.SetInputConnection(thresh.GetOutputPort())
                surf.Update()

                dec = vtk.vtkQuadricDecimation()
                dec.SetInputConnection(surf.GetOutputPort())
                dec.SetTargetReduction(0.4)
                dec.Update()

                out = dec.GetOutput()
                constLabel = vtk.vtkIntArray()
                constLabel.SetName("Label")
                constLabel.SetNumberOfComponents(1)
                constLabel.SetNumberOfTuples(out.GetNumberOfCells())
                for c in range(out.GetNumberOfCells()):
                    constLabel.SetValue(c, int(labelValue))
                out.GetCellData().AddArray(constLabel)
                out.GetCellData().SetScalars(constLabel)

                append.AddInputData(out)

            append.Update()
            self.log_info("MergedVTK: AppendPolyData done")

            # Transform + Write
            self.log_info("MergedVTK: Transform & Write")
            ijk2ras = vtk.vtkMatrix4x4()
            labelmapVolumeNode.GetIJKToRASMatrix(ijk2ras)
            parentMat = vtk.vtkMatrix4x4()
            parentMat.Identity()
            
            if self.currentVolumeNode and self.currentVolumeNode.GetParentTransformNode():
                self.currentVolumeNode.GetParentTransformNode().GetMatrixTransformToWorld(parentMat)
            
            rasMat = vtk.vtkMatrix4x4()
            vtk.vtkMatrix4x4.Multiply4x4(parentMat, ijk2ras, rasMat)

            rasT = vtk.vtkTransform()
            rasT.SetMatrix(rasMat)
            rasF = vtk.vtkTransformPolyDataFilter()
            rasF.SetTransform(rasT)
            rasF.SetInputConnection(append.GetOutputPort())
            rasF.Update()
            
            lpsT = vtk.vtkTransform()
            lpsT.Scale(-1, -1, 1)
            lpsF = vtk.vtkTransformPolyDataFilter()
            lpsF.SetTransform(lpsT)
            lpsF.SetInputConnection(rasF.GetOutputPort())
            lpsF.Update()

            outPath = os.path.join(self.outputFolderPath, f"{segmentationNode.GetName()}_merged.vtk")
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(outPath)
            writer.SetInputData(lpsF.GetOutput())
            writer.SetFileTypeToBinary()
            writer.Write()
            
            slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
            self.log_info(f"MergedVTK saved to {outPath}")

        except Exception as e:
            self.log_error(f"Error exporting MergedVTK: {str(e)}")
    
    def _exportVTKPerLabel(self, segmentationNode):
        """Export VTK par label - un fichier VTK par segment"""
        try:
            import os, re
            
            self.log_info("PerLabelVTK: Start")
            
            segmentationNode.CreateClosedSurfaceRepresentation()
            segmentation = segmentationNode.GetSegmentation()
            segSafe = re.sub(r"[^0-9A-Za-z_-]+", "_", segmentationNode.GetName())
            
            tr = segmentationNode.GetParentTransformNode()
            parentMat = vtk.vtkMatrix4x4()
            parentMat.Identity()
            if tr:
                tr.GetMatrixTransformToWorld(parentMat)

            segmentIDs = segmentation.GetSegmentIDs()
            total = len(segmentIDs)
            
            for idx, segId in enumerate(segmentIDs, start=1):
                self.log_info(f"PerLabelVTK: Segment {idx}/{total}")
                
                slicer.app.processEvents()

                segment = segmentation.GetSegment(segId)
                poly = segment.GetRepresentation("Closed surface")
                if not poly or poly.GetNumberOfPoints() == 0:
                    continue

                # Clean + smooth
                clean = vtk.vtkCleanPolyData()
                clean.SetInputData(poly)
                clean.Update()
                
                ws = vtk.vtkWindowedSincPolyDataFilter()
                ws.SetInputConnection(clean.GetOutputPort())
                ws.SetNumberOfIterations(60)
                ws.SetPassBand(0.05)
                ws.BoundarySmoothingOn()
                ws.FeatureEdgeSmoothingOn()
                ws.NonManifoldSmoothingOn()
                ws.NormalizeCoordinatesOn()
                ws.Update()

                # Normales
                flatN = vtk.vtkPolyDataNormals()
                flatN.SetInputConnection(ws.GetOutputPort())
                flatN.ComputePointNormalsOff()
                flatN.ComputeCellNormalsOn()
                flatN.SplittingOff()
                flatN.AutoOrientNormalsOn()
                flatN.ConsistencyOn()
                flatN.SetFeatureAngle(180)
                flatN.Update()

                # Decimation
                self.log_info(f"PerLabelVTK: Decimating {segment.GetName()}")
                dec = vtk.vtkQuadricDecimation()
                dec.SetInputConnection(flatN.GetOutputPort())
                dec.SetTargetReduction(0.4)
                dec.Update()

                # Transform & Write
                rasT = vtk.vtkTransform()
                rasT.SetMatrix(parentMat)
                rasF = vtk.vtkTransformPolyDataFilter()
                rasF.SetTransform(rasT)
                rasF.SetInputConnection(dec.GetOutputPort())
                rasF.Update()
                
                lpsT = vtk.vtkTransform()
                lpsT.Scale(-1, -1, 1)
                lpsF = vtk.vtkTransformPolyDataFilter()
                lpsF.SetTransform(lpsT)
                lpsF.SetInputConnection(rasF.GetOutputPort())
                lpsF.Update()

                labelSafe = re.sub(r"[^0-9A-Za-z_-]+", "_", segment.GetName())
                outPath = os.path.join(self.outputFolderPath, f"{segmentationNode.GetName()}_{labelSafe}.vtk")
                self.log_info(f"PerLabelVTK: Writing {labelSafe}.vtk")
                
                writer = vtk.vtkPolyDataWriter()
                writer.SetFileName(outPath)
                writer.SetInputData(lpsF.GetOutput())
                writer.SetFileTypeToBinary()
                writer.Write()

            self.log_info("PerLabelVTK: Done")

        except Exception as e:
            self.log_error(f"Error exporting VTKPerLabel: {str(e)}")
    
    def _cleanupAfterCase(self, volumeNode, segmentationNode):
        """Cleanup after each case"""
        try:
            self.log_info("Starting cleanup")
            
            # Suppression des nodes
            if segmentationNode and slicer.mrmlScene.IsNodePresent(segmentationNode):
                slicer.mrmlScene.RemoveNode(segmentationNode)
            
            if volumeNode and slicer.mrmlScene.IsNodePresent(volumeNode):
                slicer.mrmlScene.RemoveNode(volumeNode)
            
            # Clean CUDA memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.log_info("CUDA cache cleared")
            except ImportError:
                pass
            
            # Garbage collection
            import gc
            gc.collect()
            
            self.log_info("Cleanup completed")
            
        except Exception as e:
            self.log_error(f"Cleanup error: {str(e)}")
    
    def stop(self):
        """Stop process"""
        self.log_info("Stop requested - canceling segmentation...")
        self.isStopping = True
        
        # Stop the logic
        if self.logic:
            try:
                self.logic.stopSegmentation()
                self.log_info("Segmentation logic stopped")
            except Exception as e:
                self.log_error(f"Error stopping segmentation logic: {str(e)}")
        
        #Cleaning
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.log_info("CUDA cache cleared during stop")
        except ImportError:
            pass
        
        import gc
        gc.collect()
        self.log_info("Stop completed")
    
    @staticmethod
    def isNNUNetModuleInstalled():
        """Check if nnunet is installed"""
        try:
            import SlicerNNUNetLib
            return True
        except ImportError:
            return False
    
    def _installNNUNetIfNeeded(self) -> bool:
        """Install NNUNet if necessary"""
        try:
            from SlicerNNUNetLib import InstallLogic
            logic = InstallLogic()
            logic.progressInfo.connect(self.log_info)
            return logic.setupPythonRequirements()
        except Exception as e:
            self.log_error(f"Error installing NNUNet: {str(e)}")
            return False
    
    def _createSlicerSegmentationLogic(self):
        """Create segmentation logic"""
        if not self.isNNUNetModuleInstalled():
            return None
        try:
            from SlicerNNUNetLib import SegmentationLogic
            logic = SegmentationLogic()
            logic.progressInfo.connect(self.log_info)
            logic.errorOccurred.connect(self.log_error)
            return logic
        except Exception as e:
            self.log_error(f"Error creating segmentation logic: {str(e)}")
            return None
    
    @classmethod
    def nnUnetFolder(cls) -> Path:
        """Retourne le dossier NNUNet"""
        fileDir = Path(__file__).parent
        return fileDir.joinpath("VFACE", "Resources", "ML").resolve()


# ─── Utils functions ─────────────────────────────────────────────────────

def run_dental_segmentation(input_folder, output_folder, model_name="DentalSegmentator", 
                           device="cuda", export_formats=None):
    """
    Main function to run dental segmentation
    
    Args:
        input_folder: Path to the folder containing the volumes to be processed
        output_folder: Path to the output folder
        model_name: Name of the model to use
        device: Computing device (cuda, cpu, mps)
        export_formats: Export formats (default STL + NIfTI)
    
    Returns:
        bool: True if successful, False otherwise
    """

    if export_formats is None:
        export_formats = ExportFormat.STL | ExportFormat.NIFTI
    
    # Create Logic instance
    logic = SegmentationLogic()
    
    try:
        # Configuration
        logic.setInputFolder(input_folder)
        logic.setOutputFolder(output_folder)
        logic.setModel(model_name)
        logic.setDevice(device)
        logic.setExportFormats(export_formats)
        
        # Traitement de tous les fichiers
        success = logic.processAllFiles()
        
        return success
        
    except Exception as e:
        logic.log_error(f"Error in run_dental_segmentation: {str(e)}")
        return False
    
    finally:
        # Nettoyage
        logic.stop()
