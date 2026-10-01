import vtk
import numpy as np
import slicer
import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from enum import Flag, auto
import qt

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger
import gc
import time
from ADTLib.model_registry import NASOMAXILLA_DENT_SEG, PEDIATRIC_DENTAL_SEG, UNIVERSAL_LAB

logger = get_logger("VFACE_segmentation_logic")

# Turn the VTK warnings off
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

def nnUnetFolder() -> Path:
    """Folder holding the nnUNet weights shipped with the module."""
    return Path(__file__).parent.parent.joinpath("Resources", "ML").resolve()


class PythonDependencyChecker:
    """Download the DentalSegmentator weights when they are missing.

    Only dataset.json and plans.json are committed: the checkpoint is far too
    large for the repository, and .gitignore excludes it. This class used to
    report "Weights check completed" without checking anything, so nnUNet was
    handed a model folder with no fold_0, refused to start, and the caller then
    waited out its full one hour timeout on a process that was never launched.
    """

    # Relative to the weights folder, the file that proves they are installed.
    CHECKPOINT = Path(
        "Dataset111_453CT", "nnUNetTrainer__nnUNetPlans__3d_fullres", "fold_0", "checkpoint_final.pth"
    )

    def __init__(self, weightsFolder=None):
        self.weightsFolder = Path(weightsFolder) if weightsFolder else nnUnetFolder()

    def areWeightsMissing(self) -> bool:
        return not self.weightsFolder.joinpath(self.CHECKPOINT).is_file()

    def downloadUrl(self):
        """The URL recorded in download_info.json, or None if unusable."""
        info_path = self.weightsFolder.joinpath("download_info.json")
        try:
            with open(info_path, encoding="utf-8") as f:
                return json.load(f).get("download_url")
        except (OSError, ValueError) as e:
            logger.error(f"Cannot read {info_path}: {e}")
            return None

    def downloadWeightsIfNeeded(self, on_line):
        """Check and download the weights if necessary"""
        if not self.areWeightsMissing():
            on_line("Weights check completed")
            return True

        url = self.downloadUrl()
        if not url:
            on_line(f"Model weights are missing and no download URL is available in {self.weightsFolder}")
            return False

        on_line(f"Model weights are missing, downloading them from {url}")
        on_line("This is about 220 MB and only happens once.")

        temp_dir = Path(slicer.util.tempDirectory())
        zip_path = temp_dir.joinpath("weights.zip")
        try:
            slicer.util.downloadFile(url, str(zip_path))
            self.weightsFolder.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as archive:
                # dataset.json and plans.json are versioned with the module, and
                # the archive ships them with CRLF line endings: extracting over
                # them leaves the checkout dirty for no change of content.
                members = [
                    m for m in archive.namelist()
                    if not self.weightsFolder.joinpath(m).exists()
                ]
                archive.extractall(self.weightsFolder, members)
        except Exception as e:
            on_line(f"Failed to download the model weights: {e}")
            return False
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        if self.areWeightsMissing():
            on_line(f"The downloaded archive did not contain {self.CHECKPOINT}")
            return False

        on_line("Weights check completed")
        return True

# ─── Segmentation Logic ───────────────────────────────────────────

class SegmentationLogic:
    """
    Class containing all the dental segmentation logic without UI
    """

    # Set once the nnUNet requirements have been resolved in this Slicer session.
    _dependenciesChecked = False

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
    
    def setInputFolder(self, folder_path):
        """Define input folder"""
        self.folderPath = folder_path
        folder = Path(folder_path)
        # Filter on the formats we handle, e.g. every NIfTI file
        self.folderFiles = [
            f for f in sorted(folder.rglob("*"))
            if f.is_file() and f.name.endswith((".nii", ".nii.gz", ".nrrd", ".nrrd.gz", ".gipl", ".gipl.gz"))
        ]
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

    # A tqdm bar: " 42%|####      | 118/280 [00:02<00:03, 45.29it/s]"
    _PROGRESS_LINE = re.compile(r"\d+%\|")

    def logInferenceOutput(self, message):
        """Log nnUNet's output, minus its progress bars.

        Qt delivers this from inside slicer.app.processEvents(), and the logger
        writes to a stdout that Slicer captures into a pipe it drains from that
        same event loop. Echoing a tqdm bar redrawn dozens of times per scan
        fills the pipe while the loop is busy in this very handler: write()
        blocks, nothing can drain the pipe any more, and Slicer freezes for good.
        """
        for line in str(message).splitlines():
            line = line.strip()
            if line and not self._PROGRESS_LINE.search(line):
                self.log_info(line)
    
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
        processed = 0
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
            
            processed += 1
            slicer.app.processEvents()

        self.log_info(f"Processing completed: {processed}/{len(self.folderFiles)} file(s) segmented")

        # Returning True regardless meant the caller logged "completed
        # successfully" and moved on to steps reading an empty output folder.
        if processed == 0:
            self.log_error("No file could be segmented")
            return False
        return True
    
    def processFile(self, file_path):
        """Process a single file"""
        try:
            #Load volume
            loaded_volume = slicer.util.loadVolume(str(file_path))
            if not loaded_volume:
                self.log_error(f"Failed to load volume: {file_path}")
                return False
            
            self.currentVolumeNode = loaded_volume
            self.log_info(f"Loaded volume: {loaded_volume.GetName()}")
            
            # Showing the volume is a courtesy, not a step of the segmentation:
            # it needs a main window, and `resetSliceViews` raised
            # "'NoneType' object has no attribute 'resetSliceViews'" without one.
            # The exception came out of processFile as a plain False, so the
            # segmentation never ran -- five times in a row -- and VFACE's
            # Heatmaps and VTK Files folders stayed empty with nothing said.
            # Measured on 2026-09-29, VFACE at run level.
            try:
                slicer.util.setSliceViewerLayers(background=loaded_volume)
                slicer.util.resetSliceViews()
            except Exception as display_error:
                self.log_info("Not showing the volume (%s); the segmentation "
                              "does not depend on it" % display_error)

            # Run segmentation
            success = self._runSegmentationForVolume(loaded_volume)
            
            return success
            
        except Exception as e:
            self.log_error(f"Error in processFile: {str(e)}")
            return False
    
    def _installDependencies(self):
        """Install the dependencies"""
        try:
            if SegmentationLogic._dependenciesChecked:
                return True

            self.log_info("Checking dependencies...")

            if not self.isNNUNetModuleInstalled():
                self.log_error("NNUNet module not installed")
                return False

            if not self._installNNUNetIfNeeded():
                return False

            if not self._dependencyChecker.downloadWeightsIfNeeded(self.log_info):
                return False

            # run_bds is called once per folder, five times per pipeline: resolving
            # the pip requirements again each time costs minutes and finds nothing.
            SegmentationLogic._dependenciesChecked = True
            self.log_info("Dependencies check completed")
            return True

        except Exception as e:
            self.log_error(f"Error installing dependencies: {str(e)}")
            return False
    
    def _runSegmentationForVolume(self, volume_node):
        """Run the segmentation"""
        try:
            
            #Model Configuration
            parameter = self._getModelParameter()
            
            if not parameter.isSelectedDeviceAvailable():
                self.log_info(f"Selected device ({parameter.device.upper()}) not available, falling back to CPU")
            
            slicer.app.processEvents()
            self.logic.setParameter(parameter)
            
            #Start the segmentation
            self.logic.startSegmentation(volume_node)

            # startSegmentation reports an invalid configuration through
            # errorOccurred and returns without launching anything. Both return
            # values used to be discarded, so the wait below polled a process
            # that would never run until its one hour timeout expired.
            process = self._inferenceProcess()
            if process is not None and process.state() == qt.QProcess.NotRunning:
                self.log_error("nnUNet did not start, see the error above")
                return False

            # Wait end of segmentation
            if not self._waitForSegmentationWithEvents():
                return False
            
            # Process results
            return self._processSegmentationResults(volume_node)
            
        except Exception as e:
            self.log_error(f"Error in segmentation: {str(e)}")
            return False
    
    def _inferenceProcess(self):
        """The QProcess running nnUNet, or None when the logic exposes no such process."""
        try:
            return self.logic.inferenceProcess.process
        except AttributeError:
            return None

    def _waitForSegmentationWithEvents(self):
        """Wait the end of the segmentation"""
        
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
                    # SlicerNNUNetLib exposes none of the above: watch the inference
                    # QProcess itself. Loading the result here instead would read the
                    # file while nnUNet is still writing it and leak a node per scan,
                    # since _processSegmentationResults loads it again right after.
                    process = self._inferenceProcess()
                    if process is None:
                        self.log_error("Cannot tell whether the segmentation is running, giving up")
                        return False
                    segmentation_finished = process.state() == qt.QProcess.NotRunning

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
            base_path = Path(__file__).parent.joinpath("Resources", "ML", "Dataset001_380CT", "nnUNetTrainer__nnUNetPlans__3d_fullres").resolve()
            self._downloadModelIfNeeded("pediatricdentalseg", base_path)
            
        elif self.selectedModel == "NasoMaxillaDentSeg":
            base_path = Path(__file__).parent.joinpath("Resources", "ML", "Dataset001_max4", "nnUNetTrainer__nnUNetPlans__3d_fullres").resolve()
            self._downloadModelIfNeeded("nasomaxilladentseg", base_path)
            
        elif self.selectedModel == "UniversalLabDentalsegmentator":
            base_path = Path(__file__).parent.joinpath("Resources", "ML", "Dataset002_380CT", "nnUNetTrainer__nnUNetPlans__3d_fullres").resolve()
            self._downloadModelIfNeeded("universallab", base_path)
            
        else:  # Default DentalSegmentator
            self.log_info("Using Dataset111_453CT for DentalSegmentator")
            base_path = Path(__file__).parent.parent.joinpath("Resources", "ML", "Dataset111_453CT", "nnUNetTrainer__nnUNetPlans__3d_fullres").resolve()
        
        return Parameter(folds="0", modelPath=base_path, device=self.selectedDevice)
    
    def _downloadModelIfNeeded(self, model_type, basePath):
        """Download the model"""
        fold_path = basePath.joinpath("fold_0")
        fold_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = fold_path.joinpath("checkpoint_final.pth")
        
        if not checkpoint.exists():
            self.log_info(f"Downloading {model_type} model...")
            
            urls = {
                "pediatricdentalseg": {
                    "checkpoint": f"{PEDIATRIC_DENTAL_SEG}/checkpoint_final.pth",
                    "dataset": f"{PEDIATRIC_DENTAL_SEG}/dataset.json",
                    "plans": f"{PEDIATRIC_DENTAL_SEG}/plans.json"
                },
                "nasomaxilladentseg": {
                    "checkpoint": f"{NASOMAXILLA_DENT_SEG}/checkpoint_final.pth",
                    "dataset": f"{NASOMAXILLA_DENT_SEG}/dataset.json",
                    "plans": f"{NASOMAXILLA_DENT_SEG}/plans.json"
                },
                "universallab": {
                    "checkpoint": f"{UNIVERSAL_LAB}/checkpoint_final.pth",
                    "dataset": f"{UNIVERSAL_LAB}/dataset.json",
                    "plans": f"{UNIVERSAL_LAB}/plans.json"
                }
            }
            
            model_urls = urls.get(model_type)
            if model_urls:
                slicer.util.downloadFile(model_urls["checkpoint"], str(checkpoint))
                slicer.util.downloadFile(model_urls["dataset"], str(basePath.joinpath("dataset.json")))
                slicer.util.downloadFile(model_urls["plans"], str(basePath.joinpath("plans.json")))
    
    def _processSegmentationResults(self, volume_node):
        """Process segmentation results"""
        try:
            #Load results
            segmentation_node = self._loadSegmentationResults()
            if not segmentation_node:
                self.log_error("No segmentation results found")
                return False
            
            segmentation_node.SetName(volume_node.GetName() + "_Segmentation")
            
            # Display progress
            self._updateSegmentationDisplay(segmentation_node)
            
            slicer.app.processEvents()
            
            # Export selected formats
            if self.exportFormats & ExportFormat.NIFTI:
                self.log_info("Starting NIfTI export...")
                self._saveSegmentationAsNifti(segmentation_node, volume_node)
                slicer.app.processEvents()
            
            if self.exportFormats & ExportFormat.STL:
                self.log_info("Starting STL export...")
                self._exportSTL(segmentation_node)
                slicer.app.processEvents()
            
            if self.exportFormats & ExportFormat.OBJ:
                self.log_info("Starting OBJ export...")
                self._exportOBJ(segmentation_node)
                slicer.app.processEvents()
            
            if self.exportFormats & ExportFormat.VTK_MERGED:
                self.log_info("Starting merged VTK export...")
                self._exportMergedVTK(segmentation_node)
                slicer.app.processEvents()
            
            if self.exportFormats & ExportFormat.VTK:
                self.log_info("Starting per-label VTK export...")
                self._exportVTKPerLabel(segmentation_node)
                slicer.app.processEvents()
            
            # Cleaning
            self._cleanupAfterCase(volume_node, segmentation_node)
            
            return True
            
        except Exception as e:
            self.log_error(f"Error processing results: {str(e)}")
            return False
    
    def _loadSegmentationResults(self):
        """Load segmentation results"""
        try:
            segmentation_node = self.logic.loadSegmentation()
            return segmentation_node
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
        
        # Apply the labels
        segment_ids = list(segmentation.GetSegmentIDs())
        for i, (segment_id, label) in enumerate(zip(segment_ids, labels)):
            segment = segmentation.GetSegment(segment_id)
            if segment:
                segment.SetName(label)
    
    def _saveSegmentationAsNifti(self, segmentationNode, volume_node):
        """Save the segmentation in the NIfTI format"""
        try:
            self.log_info("Saving segmentation as NIfTI")
            
            if volume_node:
                segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)
            
            labelmap_volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            success = slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
                segmentationNode, labelmap_volume_node, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY)
            
            if not success:
                self.log_error("Failed to export segments to labelmap")
                return False
            
            output_path = os.path.join(self.outputFolderPath, segmentationNode.GetName() + ".nii.gz")
            saved = slicer.util.saveNode(labelmap_volume_node, output_path)
            
            if saved:
                self.log_info(f"Segmentation saved to {output_path}")
            else:
                self.log_error(f"Failed to save segmentation to {output_path}")
            
            # Nettoyage du label-map temporaire
            slicer.mrmlScene.RemoveNode(labelmap_volume_node)
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
            import os
            from vtk.util.numpy_support import vtk_to_numpy
            
            self.log_info("MergedVTK: Start")
            
            # Create labelmap
            labelmap_volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segmentationNode, labelmap_volume_node)
            img = labelmap_volume_node.GetImageData()

            # Marching Cubes
            self.log_info("MergedVTK: MarchingCubes")
            mc = vtk.vtkDiscreteMarchingCubes()
            mc.SetInputData(img)
            # SetValue takes a contour index, not a label value: indexing by label
            # leaves index 0 at its default and meshes the background as well.
            label_values = [int(l) for l in np.unique(vtk_to_numpy(img.GetPointData().GetScalars())) if l]
            mc.SetNumberOfContours(len(label_values))
            for i, l in enumerate(label_values):
                mc.SetValue(i, l)
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
            flat_n = vtk.vtkPolyDataNormals()
            flat_n.SetInputConnection(ws.GetOutputPort())
            flat_n.ComputePointNormalsOff()
            flat_n.ComputeCellNormalsOn()
            flat_n.SplittingOff()
            flat_n.AutoOrientNormalsOn()
            flat_n.ConsistencyOn()
            flat_n.SetFeatureAngle(180)
            flat_n.Update()

            raw_poly = flat_n.GetOutput()
            label_array = raw_poly.GetCellData().GetScalars()
            labels = np.unique(vtk_to_numpy(label_array))
            append = vtk.vtkAppendPolyData()

            # Walk the labels
            for i, label_value in enumerate(labels, start=1):
                if label_value == 0:
                    continue
                self.log_info(f"MergedVTK: Processing label {int(label_value)} ({i}/{len(labels)})")
                
                slicer.app.processEvents()

                thresh = vtk.vtkThreshold()
                thresh.SetInputData(raw_poly)
                thresh.SetInputArrayToProcess(0, 0, 0,
                    vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,
                    label_array.GetName())
                thresh.SetLowerThreshold(label_value)
                thresh.SetUpperThreshold(label_value)
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
                const_label = vtk.vtkIntArray()
                const_label.SetName("Label")
                const_label.SetNumberOfComponents(1)
                const_label.SetNumberOfTuples(out.GetNumberOfCells())
                const_label.FillComponent(0, float(label_value))
                out.GetCellData().AddArray(const_label)
                out.GetCellData().SetScalars(const_label)

                append.AddInputData(out)

            append.Update()
            self.log_info("MergedVTK: AppendPolyData done")

            # Transform + Write
            self.log_info("MergedVTK: Transform & Write")
            ijk2ras = vtk.vtkMatrix4x4()
            labelmap_volume_node.GetIJKToRASMatrix(ijk2ras)
            parent_mat = vtk.vtkMatrix4x4()
            parent_mat.Identity()
            
            if self.currentVolumeNode and self.currentVolumeNode.GetParentTransformNode():
                self.currentVolumeNode.GetParentTransformNode().GetMatrixTransformToWorld(parent_mat)
            
            ras_mat = vtk.vtkMatrix4x4()
            vtk.vtkMatrix4x4.Multiply4x4(parent_mat, ijk2ras, ras_mat)

            ras_t = vtk.vtkTransform()
            ras_t.SetMatrix(ras_mat)
            ras_f = vtk.vtkTransformPolyDataFilter()
            ras_f.SetTransform(ras_t)
            ras_f.SetInputConnection(append.GetOutputPort())
            ras_f.Update()
            
            lps_t = vtk.vtkTransform()
            lps_t.Scale(-1, -1, 1)
            lps_f = vtk.vtkTransformPolyDataFilter()
            lps_f.SetTransform(lps_t)
            lps_f.SetInputConnection(ras_f.GetOutputPort())
            lps_f.Update()

            out_path = os.path.join(self.outputFolderPath, f"{segmentationNode.GetName()}_merged.vtk")
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(out_path)
            writer.SetInputData(lps_f.GetOutput())
            writer.SetFileTypeToBinary()
            writer.Write()
            
            slicer.mrmlScene.RemoveNode(labelmap_volume_node)
            self.log_info(f"MergedVTK saved to {out_path}")

        except Exception as e:
            self.log_error(f"Error exporting MergedVTK: {str(e)}")
    
    def _exportVTKPerLabel(self, segmentationNode):
        """Export VTK per label - one VTK file per segment"""
        try:
            import os, re
            
            self.log_info("PerLabelVTK: Start")
            
            segmentationNode.CreateClosedSurfaceRepresentation()
            segmentation = segmentationNode.GetSegmentation()
            seg_safe = re.sub(r"[^0-9A-Za-z_-]+", "_", segmentationNode.GetName())
            
            tr = segmentationNode.GetParentTransformNode()
            parent_mat = vtk.vtkMatrix4x4()
            parent_mat.Identity()
            if tr:
                tr.GetMatrixTransformToWorld(parent_mat)

            segment_i_ds = segmentation.GetSegmentIDs()
            total = len(segment_i_ds)
            
            for idx, seg_id in enumerate(segment_i_ds, start=1):
                self.log_info(f"PerLabelVTK: Segment {idx}/{total}")
                
                slicer.app.processEvents()

                segment = segmentation.GetSegment(seg_id)
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
                flat_n = vtk.vtkPolyDataNormals()
                flat_n.SetInputConnection(ws.GetOutputPort())
                flat_n.ComputePointNormalsOff()
                flat_n.ComputeCellNormalsOn()
                flat_n.SplittingOff()
                flat_n.AutoOrientNormalsOn()
                flat_n.ConsistencyOn()
                flat_n.SetFeatureAngle(180)
                flat_n.Update()

                # Decimation
                self.log_info(f"PerLabelVTK: Decimating {segment.GetName()}")
                dec = vtk.vtkQuadricDecimation()
                dec.SetInputConnection(flat_n.GetOutputPort())
                dec.SetTargetReduction(0.4)
                dec.Update()

                # Transform & Write
                ras_t = vtk.vtkTransform()
                ras_t.SetMatrix(parent_mat)
                ras_f = vtk.vtkTransformPolyDataFilter()
                ras_f.SetTransform(ras_t)
                ras_f.SetInputConnection(dec.GetOutputPort())
                ras_f.Update()
                
                lps_t = vtk.vtkTransform()
                lps_t.Scale(-1, -1, 1)
                lps_f = vtk.vtkTransformPolyDataFilter()
                lps_f.SetTransform(lps_t)
                lps_f.SetInputConnection(ras_f.GetOutputPort())
                lps_f.Update()

                label_safe = re.sub(r"[^0-9A-Za-z_-]+", "_", segment.GetName())
                out_path = os.path.join(self.outputFolderPath, f"{segmentationNode.GetName()}_{label_safe}.vtk")
                self.log_info(f"PerLabelVTK: Writing {label_safe}.vtk")
                
                writer = vtk.vtkPolyDataWriter()
                writer.SetFileName(out_path)
                writer.SetInputData(lps_f.GetOutput())
                writer.SetFileTypeToBinary()
                writer.Write()

            self.log_info("PerLabelVTK: Done")

        except Exception as e:
            self.log_error(f"Error exporting VTKPerLabel: {str(e)}")
    
    def _cleanupAfterCase(self, volume_node, segmentationNode):
        """Cleanup after each case"""
        try:
            self.log_info("Starting cleanup")
            
            # Remove the nodes
            if segmentationNode and slicer.mrmlScene.IsNodePresent(segmentationNode):
                slicer.mrmlScene.RemoveNode(segmentationNode)
            
            if volume_node and slicer.mrmlScene.IsNodePresent(volume_node):
                slicer.mrmlScene.RemoveNode(volume_node)
            
            # Clean CUDA memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.log_info("CUDA cache cleared")
            except ImportError:
                pass
            
            # Garbage collection
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
        
        gc.collect()
        self.log_info("Stop completed")
    
    @staticmethod
    def isNNUNetModuleInstalled():
        """Check if nnunet is installed"""
        try:
            import SlicerNNUNetLib  # noqa: F401  (sonde de disponibilite)
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
            logic.progressInfo.connect(self.logInferenceOutput)
            logic.errorOccurred.connect(self.log_error)
            return logic
        except Exception as e:
            self.log_error(f"Error creating segmentation logic: {str(e)}")
            return None
    
    @classmethod
    def nnUnetFolder(cls) -> Path:
        """Return the nnUNet folder"""
        # This used to build <...>/VFACE_utils/VFACE/Resources/ML, which does not exist.
        return nnUnetFolder()


# ─── Utils functions ─────────────────────────────────────────────────────

# The SegmentationLogic currently running, so Cancel can reach it. Slicer stays
# responsive during segmentation (processEvents is called between files and while
# waiting on nnUNet), but a Cancel handler that built a fresh SegmentationLogic
# was stopping a brand new idle process instead of the running one.
_activeLogic = None


def stop_active_segmentation():
    """Stop the segmentation currently running, if any."""
    if _activeLogic is None:
        return False
    _activeLogic.stop()
    return True


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
    global _activeLogic
    logic = SegmentationLogic()
    _activeLogic = logic

    try:
        # Configuration
        logic.setInputFolder(input_folder)
        logic.setOutputFolder(output_folder)
        logic.setModel(model_name)
        logic.setDevice(device)
        logic.setExportFormats(export_formats)
        
        # Process every file
        success = logic.processAllFiles()
        
        return success
        
    except Exception as e:
        logic.log_error(f"Error in run_dental_segmentation: {str(e)}")
        return False
    
    finally:
        # Nettoyage
        logic.stop()
        _activeLogic = None
