# -*- coding: utf-8 -*-
"""
3 D Slicer 5 – Mask‑R‑CNN CBCT segmentation (GPU/CPU)
2025-07-14 → patch 2025-07-15 : batch Conda, anti-dead-lock
"""

import os, sys, glob, queue, time, threading, subprocess, urllib.request
from pathlib import Path
from typing import Optional, List

# Slicer / Qt
import slicer, qt
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleWidget
from slicer.util import VTKObservationMixin
import json

# Slicer-Conda helper
from CondaSetUp import CondaSetUpCall

# ───────────────────────────────────────────────────────────────────────────
def _ui_log(q: queue.Queue, msg: str):
    q.put(("log", msg))

def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for k in ("PYTHONHOME", "PYTHONPATH"):
        env.pop(k, None)
    return env

class _SafeSignals(qt.QObject):
    progress = qt.Signal(int)
    log      = qt.Signal(str)

class CLIC(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title               = "CLIC"
        parent.categories          = ["Automated Dental Tools"]
        parent.contributors        = ["Enzo Tulissi", "Lucia Cevidanes", "Juan-Carlos Prieto"]
        parent.helpText            = "Mask‑R‑CNN CBCT segmentation (GPU/CPU)"
        parent.acknowledgementText = "Model courtesy of the community."

class CLICWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        super().__init__(parent)
        VTKObservationMixin.__init__(self)
        self.conda         = CondaSetUpCall()
        self.ui_q          = queue.Queue()
        self.input_path    = None
        self.model_dir     = None
        self.output_dir    = None
        self.currentSegNode= None
        self._env_ready    = False
        self.name_env      = "clic_env"

    def setup(self):
        super().setup()
        w = slicer.util.loadUI(self.resourcePath("UI/CLIC.ui"))
        self.layout.addWidget(w)
        self.ui = slicer.util.childWidgetVariables(w)
        w.setMRMLScene(slicer.mrmlScene)

        self.sig = _SafeSignals()
        self.sig.progress.connect(self.ui.progressBar.setValue)
        self.sig.log.connect(self.ui.logTextEdit.append)

        self.ui.SearchScanFolder.clicked.connect(
            lambda: self._browse("Select input folder", "lineEditScanPath", "input_path")
        )
        self.ui.SearchModelFolder.clicked.connect(
            lambda: self._browse("Select model folder", "lineEditModelPath", "model_dir")
        )
        self.ui.SearchSaveFolder.clicked.connect(
            lambda: self._browse("Select output folder", "SaveFolderLineEdit", "output_dir")
        )
        self.ui.DownloadModelPushButton.clicked.connect(self._download_model)
        self.ui.PredictionButton.clicked.connect(self._on_predict)
        self.ui.CancelButton.clicked.connect(self._on_cancel)

        self.ui.progressBar.setVisible(False)
        self.ui.PredScanProgressBar.setVisible(False)

        # Keep legend alive
        sh = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        if sh:
            sh.AddObserver("SubjectHierarchyItemModifiedEvent", self._on_sh_modified)

    def _ensure_env(self) -> bool:
        # DEBUG: entrée dans la fonction
        print(f"[DEBUG] _ensure_env called, name_env='{self.name_env}'")

        # 1) create/test env
        exists_before = self.conda.condaTestEnv(self.name_env)
        print(f"[DEBUG] condaTestEnv('{self.name_env}') before creation → {exists_before!r}")
        if not exists_before:
            self.sig.log.emit(f"[Conda] Creating env '{self.name_env}'…")
            print(f"[DEBUG] calling condaCreateEnv({self.name_env}, '3.9', ['numpy','scipy','nibabel','requests'])")
            self.conda.condaCreateEnv(self.name_env, "3.9", ["numpy","scipy","nibabel","requests"])
            exists_after = self.conda.condaTestEnv(self.name_env)
            print(f"[DEBUG] condaTestEnv('{self.name_env}') after creation → {exists_after!r}")
            if not exists_after:
                self.sig.log.emit("❌ env creation failed")
                print("[DEBUG] env creation failed, aborting")
                return False
            self.sig.log.emit("✔ env created")
            print("[DEBUG] env created successfully")
        else:
            self.sig.log.emit(f"✔ env '{self.name_env}' exists")
            print(f"[DEBUG] env '{self.name_env}' already exists, skipping creation")

        # 2) install torch/cu118 first
        print("[DEBUG] about to install torch/cu118 via condaRunCommand")
        rc = self.conda.condaRunCommand([
            "python", "-m", "pip", "install", "--no-cache-dir",
            "--index-url=https://download.pytorch.org/whl/cu118",
            "torch==2.2.2+cu118", "torchvision==0.17.2+cu118", "torchaudio==2.2.2"
        ], self.name_env)
        self.sig.log.emit(f"[DEBUG] torch pip rc={rc!r}")
        print(f"[DEBUG] torch install returned → {rc!r}")
        if isinstance(rc, int) and rc != 0:
            self.sig.log.emit("❌ torch install failed")
            print("[DEBUG] torch install failed (int rc)")
            return False
        if isinstance(rc, str) and any(err in rc.lower() for err in ("error","failed")):
            self.sig.log.emit("❌ torch install failed")
            print("[DEBUG] torch install failed (string rc)")
            return False
        self.sig.log.emit("✔ torch installed")
        print("[DEBUG] torch installed successfully")

        # 3) downgrade numpy <2.0 for compatibility
        self.sig.log.emit("→ pip install numpy<2.0 for NumPy 1.x compatibility")
        print("[DEBUG] about to downgrade numpy with condaRunCommand")
        rc2 = self.conda.condaRunCommand([
            "python", "-m", "pip", "install", "--no-cache-dir", "'numpy<2.0'"
        ], self.name_env)
        self.sig.log.emit(f"[DEBUG] numpy downgrade rc={rc2!r}")
        print(f"[DEBUG] numpy downgrade returned → {rc2!r}")
        if isinstance(rc2, int) and rc2 != 0:
            self.sig.log.emit("❌ numpy downgrade failed")
            print("[DEBUG] numpy downgrade failed (int rc2)")
            return False
        if isinstance(rc2, str) and any(err in rc2.lower() for err in ("error","failed")):
            self.sig.log.emit("❌ numpy downgrade failed")
            print("[DEBUG] numpy downgrade failed (string rc2)")
            return False

        # env ready
        self._env_ready = True
        self.sig.progress.emit(100)
        self.sig.log.emit("✔ env ready")
        print("[DEBUG] _ensure_env succeeded, env ready")
        return True


    def _on_predict(self):
        if not self.input_path or not self.model_dir:
            qt.QMessageBox.warning(self.parent, "I/O", "Select INPUT & MODEL folders")
            return
        self._toggle_ui(True)
        self.ui.progressBar.setVisible(False)
        # update the label to show "Processing..." instead of "Downloading..."
        self.ui.PredScanLabel.setText("Processing …")
        self.ui.PredScanLabel.setVisible(True)
        if not self._ensure_env():
            self._toggle_ui(False)
            return
        scans = self._collect_scans(self.input_path)
        if not scans:
            qt.QMessageBox.warning(self.parent, "Input", "No scan found.")
            self._toggle_ui(False)
            return
        cancel_evt = threading.Event()
        t0 = time.time()

        def worker():
            for idx, scan in enumerate(scans,1):
                if cancel_evt.is_set(): break
                _ui_log(self.ui_q, f"===== {idx}/{len(scans)} — {scan.name} =====")
                self.ui_q.put(("loadScan", str(scan)))
                tmp = Path(slicer.app.temporaryPath)/f"clic_{idx}.json"
                tmp.write_text(json.dumps({
                    "input_path": str(scan),
                    "model_folder": self.model_dir,
                    "output_dir": self.output_dir,
                    "suffix": self.ui.suffixLineEdit.text or "seg"
                }))
                            # debug avant d’invoquer CondaSetUp
                self.sig.log.emit(f"[DEBUG] getCondaPath(): {self.conda.getCondaPath()!r}")
                self.sig.log.emit(f"[DEBUG] conda executable: {self.conda.getCondaExecutable()!r}")
                self.sig.log.emit(f"[DEBUG] condaTestEnv('{self.name_env}') → {self.conda.condaTestEnv(self.name_env)}")

                out = self.conda.condaRunFilePython(
                    str(Path(__file__).parent/"runner"/"clic_runner.py"),
                    [f"--params_json={tmp}"],
                    self.name_env
                )
                            # debug sortie brute
                self.sig.log.emit(f"[DEBUG] condaRunFilePython output: {out!r}")
                for ln in str(out).splitlines():
                    if ln.startswith("[PROGRESS]"):
                        self.ui_q.put(("progress", int(ln.split()[1])))
                    elif ln.startswith("[SEG]"):
                        self.ui_q.put(("segmentation", ln.split(maxsplit=1)[1]))
                    else:
                        self.ui_q.put(("log", ln))
                tmp.unlink(missing_ok=True)
            cancel_evt.set()

        threading.Thread(target=worker, daemon=True).start()
        while not cancel_evt.wait(0.05):
            slicer.app.processEvents()
            self._flush_q()
            self.ui.TimerLabel.setText(f"{time.time()-t0:.1f}s")
        self._flush_q()
        self.sig.log.emit("✔ ALL DONE")
        self.ui.PredScanLabel.setText("Done")
        self._toggle_ui(False)

    def _on_cancel(self):
        self.sig.log.emit("[User] Cancel requested.")

    def _toggle_ui(self, busy: bool):
        self.ui.progressBar.setVisible(busy)
        self.ui.PredictionButton.setEnabled(not busy)
        self.ui.CancelButton.setEnabled(busy)

    def _flush_q(self):
        while not self.ui_q.empty():
            act, data = self.ui_q.get()
            if act == "progress":
                self.ui.progressBar.setValue(int(data))
            elif act == "log":
                self.ui.logTextEdit.append(data)
            elif act == "loadScan":
                slicer.util.loadVolume(data)
            elif act == "segmentation":
                seg = slicer.util.loadSegmentation(data)
                self.currentSegNode = seg
                self._legend()

    def _browse(self, caption, le_name, attr):
        p = qt.QFileDialog.getExistingDirectory(self.parent, caption)
        if p:
            setattr(self, attr, p)
            getattr(self.ui, le_name).setText(p)

    def _collect_scans(self, root) -> List[Path]:
        p = Path(root)
        exts = (".nii", ".nii.gz", ".nrrd", ".mha", ".mhd")
        if p.is_dir():
            dcm = [d for d in p.iterdir() if d.is_dir() and any(f.suffix.lower() in exts for f in d.iterdir())]
            return sorted(dcm) if dcm else sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
        return [p]

    def _download_model(self):
        import requests
        url = (
            "https://github.com/DCBIA-OrthoLab/"
            "SlicerAutomatedDentalTools/releases/download/"
            "CLIC_model/final_model.pth"
        )
        dst = Path.home()/"Documents"/"CLIC_Models"
        dst.mkdir(exist_ok=True)
        out = dst/"final_model.pth"
        try:
            self.ui.PredScanLabel.setText("Downloading …")
            self.ui.PredScanProgressBar.setVisible(True)
            r = requests.get(url, stream=True, timeout=120); r.raise_for_status()
            tot = int(r.headers.get("content-length", 0)); done = 0
            with open(out, "wb") as f:
                for chunk in r.iter_content(1<<20):
                    f.write(chunk); done += len(chunk)
                    if tot: self.ui.PredScanProgressBar.setValue(done*100//tot)
                    slicer.app.processEvents()
            self.model_dir = str(dst)
            self.ui.lineEditModelPath.setText(self.model_dir)
            _ui_log(self.ui_q, "Model downloaded ✓")
        except Exception as e:
            qt.QMessageBox.warning(self.parent, "Download", str(e))
            _ui_log(self.ui_q, f"[ERROR] {e}")
        finally:
            time.sleep(0.4)
            self.ui.PredScanProgressBar.setVisible(False)

    def _legend(self):
        import vtk
        if not self.currentSegNode: return
        names = {1: "Buccal", 2: "Bicortical", 3: "Palatal"}
        cols  = {1: (0,1,0), 2: (1,1,0), 3: (.6,.4,.2)}
        seg = self.currentSegNode.GetSegmentation()
        ids = vtk.vtkStringArray(); seg.GetSegmentIDs(ids)
        for i in range(ids.GetNumberOfValues()):
            seg.GetSegment(ids.GetValue(i)).SetColor(*cols.get(i+1,(1,1,1)))
        lm = slicer.app.layoutManager()
        for vn in ("Red","Yellow","Green"):  
            try:
                view = lm.sliceWidget(vn).sliceView()
                ren  = view.renderWindow().GetRenderers().GetFirstRenderer()
                for a in list(ren.GetActors2D()):
                    if getattr(a,"_leg",False): ren.RemoveActor(a)
                y0,dy,fs = 0.85,0.06,16
                for k,l in names.items():
                    t = vtk.vtkTextActor(); t._leg=True; t.SetInput("■ "+l)
                    tp = t.GetTextProperty(); tp.SetFontSize(fs); tp.SetColor(*cols[k]); tp.BoldOn()
                    t.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
                    t.SetPosition(0.77, y0-(k-1)*dy); ren.AddActor2D(t)
                view.forceRender()
            except Exception:
                pass

    def _on_sh_modified(self, caller, event):
        sh = caller; nid = sh.GetActiveItemID()
        if nid:
            n = sh.GetItemDataNode(nid)
            if n and n.IsA("vtkMRMLSegmentationNode"):
                self.currentSegNode = n; self._legend()

    def initializeParameterNode(self):
        pass
