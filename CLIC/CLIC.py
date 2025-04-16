import os
import slicer
import qt
import webbrowser
import ctk
from slicer.ScriptedLoadableModule import *

# --- Les imports externes suivants sont remplacés par un lazy import,
#     afin de permettre le chargement du module même si ces librairies ne sont pas installées ---
# import torch
# import subprocess
import numpy as np
# import nibabel as nib
# from torchvision.models.detection import maskrcnn_resnet50_fpn
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# import scipy.ndimage
# import time
# from CondaSetUp import CondaSetUpCall, CondaSetUpCallWsl
import threading
import logging
# import requests
# import glob
# import zipfile
# import queue
# import sys
# import vtk

from slicer.util import VTKObservationMixin

# --- Fonction de lazy import ---
def lazy_import(module_name, fromlist=None):
    return __import__(module_name, fromlist=fromlist)

def PathFromNode(node):
    storageNode = node.GetStorageNode()
    if storageNode is not None:
        filepath = storageNode.GetFullNameFromFileName()
    else:
        filepath = None
    return filepath

# Classe pour diffuser l'avancement et les logs
class ProgressUpdater(qt.QObject):
    progressChanged = qt.Signal(int)
    logChanged = qt.Signal(str)

class CLIC(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "CLIC"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Enzo Tulissi"]
        self.parent.helpText = """This module performs segmentation using a Mask R-CNN model."""
        self.parent.acknowledgementText = "Thanks to the community for support."

class CLICWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self.CBCT_as_input = True
        self.folder_as_input = False
        self.MRMLNode_scan = None
        self.input_path = None
        self.model_folder = None
        self.output_folder = None
        self.goup_output_files = False
        self.scan_count = 0
        # Listes pour conserver scans et segmentations
        self.scanNodes = []
        self.segmentationNodes = []
        self.ui_queue = lazy_import("queue").Queue()
        self._shObserverTag = None

    def UpdateSaveType(self, checked):
        self.goup_output_files = checked
        self.ui.SearchSaveFolder.setHidden(checked)
        self.ui.SaveFolderLineEdit.setHidden(checked)
        self.ui.PredictFolderLabel.setHidden(checked)

    def setup(self):
        # Import lazy pour CondaSetUpCallWsl
        try:
            CondaSetUpCallWsl = lazy_import("CondaSetUp", fromlist=["CondaSetUpCallWsl"]).CondaSetUpCallWsl
            self.conda_wsl = CondaSetUpCallWsl()
        except Exception as e:
            self.conda_wsl = None
            print(f"[WARN] CondaSetUp n'a pas pu être importé en lazy: {e}")
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/CLIC.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Création de la logique métier
        self.logic = CLICLogic()

        # Connecter les signaux pour la progression et les logs
        self.progressUpdater = ProgressUpdater()
        self.progressUpdater.progressChanged.connect(lambda value: self.ui.progressBar.setValue(value))
        self.progressUpdater.logChanged.connect(lambda text: self.ui.logTextEdit.append(text))

        self.ui.SavePredictCheckBox.toggled.connect(self.UpdateSaveType)
        self.ui.DownloadModelPushButton.clicked.connect(self.onModelDownloadButton)
        self.ui.SearchScanFolder.clicked.connect(self.onSearchScanButton)
        self.ui.SearchModelFolder.clicked.connect(self.onSearchModelButton)
        self.ui.SearchSaveFolder.clicked.connect(self.onSearchSaveButton)
        self.ui.PredictionButton.clicked.connect(self.onPredictButton)
        self.ui.CancelButton.clicked.connect(self.onCancel)

        self.ui.progressBar.setVisible(False)
        self.ui.PredScanProgressBar.setVisible(False)
        self.RunningUI(False)
        self.initializeParameterNode()

        # Ajout d'un observateur pour le Subject Hierarchy
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        if shNode:
            self._shObserverTag = shNode.AddObserver("SubjectHierarchyItemModifiedEvent", self.onSubjectHierarchyModified)
        else:
            print("[WARN] Impossible d'obtenir le Subject Hierarchy Node.")

    def SwitchInputExtension(self, index):
        if index == 1:
            self.ui.ScanPathLabel.setText('DICOM Folder')
        else:
            self.ui.ScanPathLabel.setText('Scan File/Folder')

    def onNodeChanged(self):
        self.MRMLNode_scan = slicer.mrmlScene.GetNodeByID(self.ui.MRMLNodeComboBox.currentText)
        if self.MRMLNode_scan:
            self.input_path = PathFromNode(self.MRMLNode_scan)
            self.scan_count = 1
            self.ui.PrePredInfo.setText("Number of scans to process: 1")
            self.load_nii_in_slicer(self.input_path)
            return True
        return False

    # Observateur pour la hiérarchie des sujets
    def onSubjectHierarchyModified(self, caller, event):
        shNode = caller
        activeItemID = shNode.GetActiveItemID()
        if activeItemID:
            activeNode = shNode.GetItemDataNode(activeItemID)
            if activeNode and activeNode.IsA("vtkMRMLSegmentationNode"):
                self.currentSegNode = activeNode
                self.attachCornerLegend(activeNode)

    def onModelDownloadButton(self):
        model_url = "https://github.com/ashmoy/maskRcnn/releases/download/model/final_model.pth"
        default_model_folder = os.path.join(os.path.expanduser("~"), "Documents", "CLIC_Models")
        os.makedirs(default_model_folder, exist_ok=True)
        model_path = os.path.join(default_model_folder, "final_model.pth")
        try:
            self.ui.PredScanLabel.setText("Downloading model...")
            self.ui.PredScanProgressBar.setVisible(True)
            # Lazy import de requests
            requests = lazy_import("requests")
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            chunk_size = 1024
            with open(model_path, 'wb') as file:
                for data in response.iter_content(chunk_size=chunk_size):
                    file.write(data)
                    downloaded_size += len(data)
                    progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                    self.ui.PredScanProgressBar.setValue(int(progress))
            self.ui.lineEditModelPath.setText(default_model_folder)
            self.model_folder = default_model_folder
            self.ui.PredScanLabel.setText("Model downloaded successfully!")
        except Exception as e:
            qt.QMessageBox.warning(self.parent, 'Error', f"Failed to download model: {str(e)}")
            self.ui.PredScanLabel.setText("Error during download.")
        finally:
            self.ui.PredScanProgressBar.setVisible(False)

    def onSearchScanButton(self):
        scan_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select a scan folder")
        if scan_folder:
            self.input_path = scan_folder
            self.ui.lineEditScanPath.setText(self.input_path)

    def onSearchModelButton(self):
        model_path = qt.QFileDialog.getExistingDirectory(self.parent, "Select Model Folder", "")
        if model_path:
            self.model_folder = model_path
            self.ui.lineEditModelPath.setText(model_path)

    def onSearchSaveButton(self):
        save_folder = qt.QFileDialog.getExistingDirectory(self.parent, "Select Save Folder", "")
        if save_folder:
            self.output_folder = save_folder
            self.ui.SaveFolderLineEdit.setText(save_folder)

    def check_dependencies(self):
        self.ui_queue.put(("log", "Checking required libraries..."))
        required_libraries = ["torch", "nibabel", "numpy", "scipy", "requests"]
        missing_libraries = []
        for lib in required_libraries:
            try:
                __import__(lib)
            except ImportError:
                missing_libraries.append(lib)
        if missing_libraries:
            self.ui_queue.put(("log", f"Missing libraries detected: {', '.join(missing_libraries)}"))
            reply = qt.QMessageBox.question(
                self.parent,
                "Missing Dependencies",
                f"The following libraries are missing: {', '.join(missing_libraries)}.\nDo you want to install them automatically?",
                qt.QMessageBox.Yes | qt.QMessageBox.No
            )
            if reply == qt.QMessageBox.Yes:
                import sys
                import subprocess
                for lib in missing_libraries:
                    try:
                        self.ui_queue.put(("log", f"Installing library: {lib}..."))
                        if lib == "torch":
                            subprocess.check_call([
                                sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio",
                                "--index-url", "https://download.pytorch.org/whl/cu118"
                            ])
                        else:
                            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
                    except Exception as e:
                        self.ui_queue.put(("log", f"Failed to install {lib}: {str(e)}"))
                        qt.QMessageBox.critical(
                            self.parent,
                            "Installation Error",
                            f"Failed to install {lib}. Error: {str(e)}"
                        )
                        return False
                for lib in missing_libraries:
                    try:
                        __import__(lib)
                    except ImportError:
                        self.ui_queue.put(("log", f"Failed to load {lib} after installation."))
                        qt.QMessageBox.critical(
                            self.parent,
                            "Dependency Error",
                            f"Failed to load {lib} after installation. Please install it manually."
                        )
                        return False
            else:
                self.ui_queue.put(("log", "User chose not to install missing libraries."))
                qt.QMessageBox.critical(
                    self.parent,
                    "Missing Dependencies",
                    "Please install the missing libraries manually before proceeding."
                )
                return False
        self.ui_queue.put(("log", "All required libraries are installed and ready to use."))
        return True

    def onPredictButton(self):
        if not self.check_dependencies():
            return
        if not self.input_path:
            qt.QMessageBox.warning(self.parent, 'Warning', 'Please select an input file/folder')
            return
        if not self.model_folder:
            qt.QMessageBox.warning(self.parent, 'Warning', 'Please select a model folder')
            return

        param = {
            "input_path": self.input_path,
            "model_folder": self.model_folder,
            "output_dir": self.output_folder,
            "suffix": self.ui.suffixLineEdit.text
        }

        def update_progress(progress):
            self.ui_queue.put(("progress", progress))

        def update_log(message):
            self.ui_queue.put(("log", message))

        def display_callback(action, file_path):
            self.ui_queue.put((action, file_path))

        self.segmentationLoadedEvent = threading.Event()

        try:
            self.RunningUI(True)
            self.processThread = threading.Thread(
                target=self.logic.process,
                args=(param, update_progress, update_log, display_callback, self.segmentationLoadedEvent)
            )
            self.processThread.start()

            import time
            start_time = time.time()
            while self.processThread.is_alive():
                slicer.app.processEvents()
                current_time = time.time()
                self.ui.TimerLabel.setText(f"Time elapsed: {current_time - start_time:.2f}s")
                self.process_ui_queue()
                time.sleep(0.1)
            self.process_ui_queue()
            self.ui_queue.put(("log", "Segmentation completed successfully!"))
        except Exception as e:
            self.ui_queue.put(("log", f"An error occurred during segmentation: {str(e)}"))
        finally:
            self.RunningUI(False)

    def process_ui_queue(self):
        while not self.ui_queue.empty():
            action, data = self.ui_queue.get()
            if action == "progress":
                self.ui.progressBar.setValue(int(data))
            elif action == "log":
                self.ui.logTextEdit.append(data)
            elif action == "loadScan":
                self.load_nii_in_slicer(data)
            elif action == "segmentation":
                self.load_segmentation(data)

    def load_segmentation(self, seg_file):
        def load_and_attach():
            segNode = slicer.util.loadSegmentation(seg_file)
            if not segNode:
                qt.QMessageBox.critical(self.parent,
                                        "Erreur de chargement",
                                        f"Échec du chargement de la segmentation pour :\n{seg_file}")
                return
            segNode.SetName(os.path.basename(seg_file))
            self.segmentationNodes.append(segNode)
            self.currentSegNode = segNode
            self.attachCornerLegend(segNode)
            if hasattr(self, 'segmentationLoadedEvent') and self.segmentationLoadedEvent:
                self.segmentationLoadedEvent.set()
        qt.QTimer.singleShot(0, load_and_attach)

    def attachCornerLegend(self, segmentationNode=None):
        import vtk
        layoutManager = slicer.app.layoutManager()
        if segmentationNode is None:
            segmentationNode = self.currentSegNode
        if not segmentationNode:
            print("[WARN] Aucun nœud de segmentation actif.")
            return
        label_names = {1: "Buccal", 2: "Bicortical", 3: "Palatal"}
        fixed_colors = {1: (0.0, 1.0, 0.0), 2: (1.0, 1.0, 0.0), 3: (0.6, 0.4, 0.2)}
        segmentation = segmentationNode.GetSegmentation()
        segmentIDs = vtk.vtkStringArray()
        segmentation.GetSegmentIDs(segmentIDs)
        for i in range(segmentIDs.GetNumberOfValues()):
            seg_id = segmentIDs.GetValue(i)
            segment = segmentation.GetSegment(seg_id)
            label_value = segment.GetLabelValue()
            if label_value in fixed_colors:
                segment.SetColor(*fixed_colors[label_value])
        for viewName in ["Red", "Yellow", "Green"]:
            try:
                view = layoutManager.sliceWidget(viewName).sliceView()
                renderer = view.renderWindow().GetRenderers().GetFirstRenderer()
                for actor in list(renderer.GetActors2D()):
                    if hasattr(actor, "_isLegendActor") and actor._isLegendActor:
                        renderer.RemoveActor(actor)
                base_y = 0.85
                spacing = 0.06
                font_size = 16
                i = 0
                for label in [1, 2, 3]:
                    name = label_names[label]
                    color = fixed_colors[label]
                    full_text = f"■ {name}"
                    text_actor = vtk.vtkTextActor()
                    text_actor.SetInput(full_text)
                    prop = text_actor.GetTextProperty()
                    prop.SetFontSize(font_size)
                    prop.SetFontFamilyToArial()
                    prop.SetColor(*color)
                    prop.BoldOn()
                    prop.ShadowOff()
                    text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
                    text_actor.SetPosition(0.77, base_y - i * spacing)
                    text_actor._isLegendActor = True
                    renderer.AddActor2D(text_actor)
                    i += 1
                view.forceRender()
            except Exception as e:
                print(f"[WARN] Could not update annotation for {viewName} view: {e}")

    def reapplyLegend(self, sliceView, text):
        try:
            renderer = sliceView.renderWindow().GetRenderers().GetFirstRenderer()
            if hasattr(sliceView, "cornerAnnotation") and sliceView.cornerAnnotation:
                sliceView.cornerAnnotation.SetText(0, text)
                renderer.AddViewProp(sliceView.cornerAnnotation)
                sliceView.forceRender()
        except Exception as e:
            print(f"[WARN] Failed to reapply legend: {e}")

    def onCancel(self):
        if self.logic:
            self.logic.cancelRequested = True
            self.progressUpdater.logChanged.emit("Cancellation requested.")
        if hasattr(self, "processThread") and self.processThread.is_alive():
            self.processThread.join(1)
            self.ui.PredScanLabel.setText("Prediction canceled.")

    def RunningUI(self, running):
        self.ui.PredictionButton.setEnabled(not running)
        self.ui.CancelButton.setEnabled(running)
        self.ui.progressBar.setVisible(running)

    def initializeParameterNode(self):
        pass

    def load_nii_in_slicer(self, nii_file):
        if not os.path.exists(nii_file):
            raise FileNotFoundError(f"NIfTI file not found: {nii_file}")
        volume_node = slicer.util.loadVolume(nii_file)
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        slicer.util.setSliceViewerLayers(background=volume_node)
        self.MRMLNode_scan = volume_node
        self.scanNodes.append(volume_node)

class CLICLogic(ScriptedLoadableModuleLogic):
    CLASS_NAMES = {1: "buccal", 2: "bicortical", 3: "palatal"}

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.seg_files = []
        self.cancelRequested = False

    def get_model_instance_segmentation(self, num_classes):
            # Lazy import des modules TorchVision requis
        maskrcnn_resnet50_fpn = lazy_import("torchvision.models.detection", fromlist=["maskrcnn_resnet50_fpn"]).maskrcnn_resnet50_fpn
        FastRCNNPredictor = lazy_import("torchvision.models.detection.faster_rcnn", fromlist=["FastRCNNPredictor"]).FastRCNNPredictor
        MaskRCNNPredictor = lazy_import("torchvision.models.detection.mask_rcnn", fromlist=["MaskRCNNPredictor"]).MaskRCNNPredictor
        model = maskrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        return model

    def load_model(self, model_path, num_classes, device):
        torch = lazy_import("torch")

        model = self.get_model_instance_segmentation(num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model

    def process_nii_file(self, model, nii_path, device, progress_callback=None, log_callback=None, score_threshold=0.7):
        scipy = lazy_import("scipy.ndimage")
        torch = lazy_import("torch")
        nib = lazy_import("nibabel")
        nib_vol = nib.load(nii_path)
        vol_data = nib_vol.get_fdata(dtype=np.float32)
        H, W, Z = vol_data.shape
        if log_callback:
            log_callback("Processing slices for file: " + nii_path)
        all_detections = []
        slice_counts = {"buccal": {"left": 0, "right": 0},
                        "bicortical": {"left": 0, "right": 0},
                        "palatal": {"left": 0, "right": 0}}
        for z in range(Z):
            if self.cancelRequested:
                if log_callback:
                    log_callback("Processing cancelled during slice processing at slice %d." % z)
                break
            slice_2d = vol_data[..., z]
            slice_norm = self.normalize_slice(slice_2d)
            slice_tensor = torch.from_numpy(slice_norm).unsqueeze(0).repeat(3, 1, 1).float().to(device)
            with torch.no_grad():
                preds = model([slice_tensor])[0]
            scores = preds["scores"]
            masks = preds.get("masks", None)
            labels = preds["labels"]
            keep = scores >= score_threshold
            scores = scores[keep]
            labels = labels[keep]
            if masks is not None:
                masks = masks[keep]
            if masks is None or len(scores) == 0:
                if progress_callback:
                    progress = ((z + 1) / Z) * 100
                    progress_callback(progress)
                continue
            labels_np = labels.cpu().numpy()
            masks_np = (masks > 0.5).squeeze(1).cpu().numpy()
            for i in range(len(masks_np)):
                l = int(labels_np[i])
                mk = masks_np[i]
                com = scipy.ndimage.center_of_mass(mk)
                side = "left" if com[0] < (H / 2) else "right"
                slice_counts[self.CLASS_NAMES[l]][side] += 1
                all_detections.append({
                    "label": l,
                    "slice_z": z,
                    "mask_2d": mk
                })
            if progress_callback:
                progress = ((z + 1) / Z) * 100
                progress_callback(progress)
        return vol_data, nib_vol, all_detections, slice_counts

    def normalize_slice(self, slice_2d):
        mn, mx = slice_2d.min(), slice_2d.max()
        if (mx - mn) > 1e-8:
            return (slice_2d - mn) / (mx - mn)
        else:
            return np.zeros_like(slice_2d, dtype=np.float32)

    def save_nii(self, vol_np, nib_ref, out_path):
        nib = lazy_import("nibabel")
        nii_img = nib.Nifti1Image(vol_np.astype(np.int16), nib_ref.affine, nib_ref.header)
        nib.save(nii_img, out_path)

    def process(self, parameters, progress_callback=None, log_callback=None, display_callback=None, sync_event=None):
        try:
            nib = lazy_import("nibabel")
            torch = lazy_import("torch")
            glob = lazy_import("glob")
            self.cancelRequested = False
            input_path = parameters["input_path"]
            model_folder = parameters["model_folder"]
            output_dir = parameters["output_dir"]
            # Récupération du suffixe avec une valeur par défaut "seg"
            suffix = parameters.get("suffix", "seg")
            suffix = suffix if suffix else "seg"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_files = glob.glob(os.path.join(model_folder, "*.pth"))
            if not model_files:
                raise FileNotFoundError("No .pth files found in the specified model folder.")
            model_path = model_files[0]
            num_classes = 4
            model = self.load_model(model_path, num_classes, device)
            if os.path.isfile(input_path) and (input_path.endswith(".nii") or input_path.endswith(".nii.gz")):
                file_list = [input_path]
            elif os.path.isdir(input_path):
                file_list = glob.glob(os.path.join(input_path, "*.nii")) + glob.glob(os.path.join(input_path, "*.nii.gz"))
            else:
                raise ValueError("Invalid input path. Please provide a valid .nii/.nii.gz file or folder.")
            seg_files = []
            for nii_file in file_list:
                if self.cancelRequested:
                    if log_callback:
                        log_callback("Processing cancelled before file: " + nii_file)
                    break
                if log_callback:
                    log_callback("Processing file: " + nii_file)
                if display_callback:
                    display_callback("loadScan", nii_file)
                vol_data, nib_ref, detections, counts = self.process_nii_file(model, nii_file, device, progress_callback, log_callback)
                if log_callback:
                    log_callback("Finished processing file: " + nii_file)
                seg_data = np.zeros(vol_data.shape, dtype=np.int16)
                for det in detections:
                    z = det["slice_z"]
                    mask = det["mask_2d"]
                    label = det["label"]
                    seg_data[..., z][mask] = label
                base_name = os.path.basename(nii_file).replace(".nii.gz", "").replace(".nii", "")
                scan_output_folder = os.path.join(output_dir, base_name)
                os.makedirs(scan_output_folder, exist_ok=True)
                output_seg_path = os.path.join(scan_output_folder, f"{base_name}_{suffix}.nii.gz")
                nii_img = nib.Nifti1Image(seg_data.astype(np.int16), nib_ref.affine, nib_ref.header)
                nii_img.header['descrip'] = str(self.CLASS_NAMES)
                nib.save(nii_img, output_seg_path)
                seg_files.append(output_seg_path)
                if display_callback:
                    display_callback("segmentation", output_seg_path)
                if sync_event is not None:
                    sync_event.wait()
                    sync_event.clear()
            if log_callback:
                log_callback("Processing completed successfully.")
            self.seg_files = seg_files
        except Exception as e:
            if log_callback:
                log_callback(f"Error during processing: {str(e)}")
            raise
