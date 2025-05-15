# CLIC.py ─────────────────────────────────────────────────────────────────────
# 3 D Slicer 5 – Mask-R-CNN CBCT segmentation (GPU/CPU)
# 2025-07-14 → patch 2025-07-15 : batch Conda, anti-dead-lock
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, glob, queue, time, threading, importlib, subprocess
from   pathlib   import Path
from   typing    import Optional, List

# Slicer / Qt
import slicer, qt
from   slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleWidget
from   slicer.util                   import VTKObservationMixin
import json
# Slicer-Conda helper
from   CondaSetUp import CondaSetUpCall
# ─────────────────────────────────────────────────────────────────────────────
def _ui_log(q:queue.Queue, msg:str): q.put(("log", msg))


class _SafeSignals(qt.QObject):            # thread-safe
    progress = qt.Signal(int)
    log      = qt.Signal(str)


# ╓────────────────────────────────────────────────────────────────────────────╖
# ║ 1. Module declaration                                                     ║
# ╙────────────────────────────────────────────────────────────────────────────╜
class CLIC(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title       = "CLIC"
        parent.categories  = ["Automated Dental Tools"]
        parent.contributors= ["Enzo Tulissi (CPE Lyon / UoM)",
                              "Lucia Cevidanes (UoM)",
                              "Juan-Carlos Prieto (UoNC)"]
        parent.helpText    = "Run Mask-R-CNN segmentation on CBCT volumes."
        parent.acknowledgementText = "Mask-R-CNN model courtesy of the community."


# ╓────────────────────────────────────────────────────────────────────────────╖
# ║ 2. Widget                                                                 ║
# ╙────────────────────────────────────────────────────────────────────────────╜
class CLICWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    # ───────────── init ──────────────────────────────────────────────────────
    def __init__(self, parent=None):
        super().__init__(parent); VTKObservationMixin.__init__(self)
        self.conda = CondaSetUpCall()
        self.ui_q  : queue.Queue = queue.Queue()
        self.logic : Optional[object] = None

        self.input_path  : Optional[str] = None
        self.model_dir   : Optional[str] = None
        self.output_dir  : Optional[str] = None

        self.currentSegNode = None
        self._env_ready = False            # devient True après création

    # ───────────── GUI setup ─────────────────────────────────────────────────
    def setup(self):
        super().setup()
        w = slicer.util.loadUI(self.resourcePath("UI/CLIC.ui"))
        self.layout.addWidget(w); self.ui = slicer.util.childWidgetVariables(w)
        w.setMRMLScene(slicer.mrmlScene)

        self.sig = _SafeSignals()
        self.sig.progress.connect(self.ui.progressBar.setValue)
        self.sig.log     .connect(self.ui.logTextEdit.append)

        # Buttons / widgets
        self.ui.SavePredictCheckBox.toggled.connect(
            lambda b: [w.setHidden(b) for w in
                       (self.ui.SearchSaveFolder,self.ui.SaveFolderLineEdit,
                        self.ui.PredictFolderLabel)])

        self.ui.SearchScanFolder .clicked.connect(
            lambda:self._browse("Select input folder","lineEditScanPath","input_path"))
        self.ui.SearchModelFolder.clicked.connect(
            lambda:self._browse("Select model folder","lineEditModelPath","model_dir"))
        self.ui.SearchSaveFolder .clicked.connect(
            lambda:self._browse("Select output folder","SaveFolderLineEdit","output_dir"))

        self.ui.DownloadModelPushButton.clicked.connect(self._download_model)
        self.ui.PredictionButton.clicked.connect(self._on_predict)
        self.ui.CancelButton.clicked.connect(self._on_cancel)

        self.ui.progressBar.setVisible(False)
        self.ui.PredScanProgressBar.setVisible(False)

        # SH observer (keep legend)
        sh = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        if sh: sh.AddObserver("SubjectHierarchyItemModifiedEvent", self._on_sh_modified)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Conda / clic_env (Python 3.9 + PyTorch GPU cu118 garanti)
# ─────────────────────────────────────────────────────────────────────────────



    def _gpu() -> bool:
        """Returns True if `nvidia-smi` detects at least one GPU."""
        try:
            subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    def _torch_cmd(self) -> List[str]:
        """
        Pip command to install torch/cu118 compatible with Python 3.9.
        """
        base = ["python", "-m", "pip", "install", "--no-cache-dir"]
        return base + [
            "torch==2.2.2+cu118",
            "torchvision==0.17.2+cu118",
            "torchaudio==2.2.2",
            "--index-url", "https://download.pytorch.org/whl/cu118",
        ]

    def _site_pkgs(self, env: str) -> Optional[str]:
        """Path to site-packages of the env 'env'."""
        root = self.conda.getCondaPath()
        if not root or root == "None":
            return None
        pattern = os.path.join(root, "envs", env, "lib", "python*", "site-packages")
        candidates = glob.glob(pattern)
        return max(candidates, key=len) if candidates else None

    def _pkg_missing(self, env: str) -> set[str]:
        """Calculates missing packages in the env (includes torch)."""
        needed = {"nibabel", "numpy", "scipy", "requests", "torch"}
        sp = self._site_pkgs(env)
        if not sp:
            return needed
        present = {p.name.split("-")[0].lower() for p in os.scandir(sp)}
        return needed - present

    def _ensure_env(self) -> bool:
        import os, subprocess, json, urllib.request, importlib

        env_name   = "clic_env"
        # Chemin Miniconda (ou emplacement par défaut si getCondaPath() renvoie None)
        conda_root = self.conda.getCondaPath() or os.path.expanduser("~/miniconda3")
        conda_exec = os.path.join(conda_root, "bin", "conda")
        conda_py   = os.path.join(conda_root, "bin", "python")
        installer_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

        # Helper pour logger et rafraîchir l'UI
        def log(msg):
            self.sig.log.emit(msg)
            slicer.app.processEvents()

        # 0) Installer Miniconda si absent
        if not os.path.isfile(conda_exec):
            log(f"Miniconda not found under {conda_root}, installation in progress...")
            tmp_installer = os.path.join(slicer.app.temporaryPath, "Miniconda3-latest.sh")

            # Téléchargement via urllib
            try:
                log("→ Downloading Miniconda…")
                urllib.request.urlretrieve(installer_url, tmp_installer)
                log("Download complete ✅")
            except Exception as e:
                log(f"Download failed : {e}")
                return False

            # Exécution de l'installateur en mode batch
            proc = subprocess.Popen(
                ["bash", tmp_installer, "-b", "-p", conda_root],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in proc.stdout:
                log(line.rstrip())
            if proc.wait() != 0:
                log("Miniconda installation failed")
                return False
            log("Miniconda installed✅")

        # Prépare un environnement épuré pour subprocess
        safe_env = os.environ.copy()
        for k in ("PYTHONPATH", "PYTHONHOME"):
            safe_env.pop(k, None)
        safe_env["PATH"] = os.path.join(conda_root, "bin") + os.pathsep + safe_env.get("PATH", "")

        # Helper pour exécuter une commande et logger stdout/stderr en temps réel
        def run_and_log(cmd, step=None, total=None):
            if step is not None and total is not None:
                pct = step * 100 // total
                self.sig.progress.emit(pct)
            try:
                p = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, env=safe_env
                )
            except Exception as e:
                log(f"Erreur lancement `{cmd[0]}` : {e}")
                return -1
            for ln in p.stdout:
                log(ln.rstrip())
            return p.wait()

        # 1) Création env Python 3.9 si nécessaire
        if not self.conda.condaTestEnv(env_name):
            log(f"[Conda] Creating env '{env_name}' (Python 3.9)…")
            if run_and_log([conda_py, "-m", "conda", "create", "-y", "-n", env_name, "python=3.9"]) != 0:
                log("Environment creation failed.")
                return False
            log("Environment created ✅")
        else:
            log(f"[Conda] The env '{env_name}' already exists.")

        # 2) Installation conda-forge
        cf_pkgs = ["nibabel", "numpy", "scipy", "requests"]
        total   = len(cf_pkgs) + 1
        for idx, pkg in enumerate(cf_pkgs, start=1):
            log(f"→ conda install {pkg}")
            if run_and_log(
                [conda_py, "-m", "conda", "install", "-y", "-n", env_name, "-c", "conda-forge", pkg],
                step=idx, total=total
            ) != 0:
                log(f"Installation failure{pkg}.")
                return False

        # 3) Installation torch/cu118 via pip
        log("→ pip install torch/cu118")
        env_py = os.path.join(conda_root, "envs", env_name, "bin", "python")
        if run_and_log(
            [
                env_py, "-m", "pip", "install", "--no-cache-dir",
                "--index-url", "https://download.pytorch.org/whl/cu118",
                "torch==2.2.2+cu118",
                "torchvision==0.17.2+cu118",
                "torchaudio==2.2.2"
            ],
            step=len(cf_pkgs)+1, total=total
        ) != 0:
            log("Torch/cu118 installation failed.")
            return False

        # 4) Ajout du site-packages et test PyTorch CUDA
        sp = self._site_pkgs(env_name)
        if sp and sp not in sys.path:
            sys.path.append(sp)
            importlib.invalidate_caches()
            log(f"+ {sp}")

        log("→ PyTorch CUDA Test…")
        if run_and_log([conda_py, "-m", "conda", "run", "-n", env_name, "python", "-c",
                        "import torch,json;print(json.dumps({'cuda':torch.cuda.is_available(),'ver':torch.__version__}))"]
        ) != 0:
            log("PyTorch CUDA test failed.")
            return False

        self.sig.progress.emit(100)
        self._env_ready = True
        log("✔ clic_env ready to use")
        return True

    # ╓─────────────────────────────────────────────────────────────────────────╖
    # ║ 4.  Prediction loop (multi-files, anti-dead-lock)                      ║
    # ╙─────────────────────────────────────────────────────────────────────────╜
    def _on_predict(self):
        if not self.input_path or not self.model_dir:
            qt.QMessageBox.warning(self.parent,"I/O","Select INPUT & MODEL folders"); return

        self._toggle_ui(True); t0=time.time()
        if not self._ensure_env():
            self._toggle_ui(False); return
        self._flush_q()

        # import logic once env is ready
        if not self.logic:
            from CLI.CLICLogic import CLICLogic
            self.logic = CLICLogic()

        # build list of scans -------------------------------------------------
        scans=[]
        inp=Path(self.input_path)
        if inp.is_dir():
            dcm_dirs=[d for d in inp.iterdir() if d.is_dir() and list(d.glob("*.dcm"))]
            if dcm_dirs: scans=sorted(dcm_dirs,key=lambda p:p.name)
            else:
                exts=(".nii",".nii.gz",".nrrd",".n5",".mha",".mhd")
                scans=sorted([p for p in inp.iterdir() if p.suffix.lower() in exts])
        else:
            scans=[inp]

        if not scans:
            qt.QMessageBox.warning(self.parent,"Input","No scan found."); self._toggle_ui(False); return

        cancel_evt=threading.Event()
        def _worker():
            for idx,scan in enumerate(scans,1):
                if cancel_evt.is_set(): break
                self.ui_q.put(("log",f"===== {idx}/{len(scans)} — {scan.name} ====="))
                params=dict(input_path=str(scan),
                            model_folder=self.model_dir,
                            output_dir=self.output_dir,
                            suffix=self.ui.suffixLineEdit.text or "seg")

                finished=threading.Event()
                # ---- lance la segmentation (condaRunCommand dans CLICLogic) ----
                self.logic.process(params,
                                   lambda p:self.ui_q.put(("progress",p)),
                                   lambda m:self.ui_q.put(("log",m)),
                                   lambda a,f:self.ui_q.put((a,f)),
                                   finished)

                # timeout 30 min / scan pour éviter blocage infini
                if not finished.wait(timeout=1800):
                    self.ui_q.put(("log",f"[TIMEOUT] {scan.name} >30 min, abort."))
                    cancel_evt.set(); break
            cancel_evt.set()

        threading.Thread(target=_worker,daemon=True).start()

        # gui loop
        while not cancel_evt.wait(0.05):
            slicer.app.processEvents(); self._flush_q()
            self.ui.TimerLabel.setText(f"{time.time()-t0:.1f}s")
        self._flush_q(); self._toggle_ui(False); _ui_log(self.ui_q,"✔ ALL DONE")

    def _on_cancel(self):
        if self.logic: self.logic.cancelRequested=True
        self.ui_q.put(("log","[User] Cancel requested."))

    # ╓─────────────────────────────────────────────────────────────────────────╖
    # ║ 5. Misc. helpers                                                       ║
    # ╙─────────────────────────────────────────────────────────────────────────╜
    def _toggle_ui(self,b):
        self.ui.progressBar.setVisible(b); self.ui.PredictionButton.setEnabled(not b)
        self.ui.CancelButton.setEnabled(b)

    def _flush_q(self):
        while not self.ui_q.empty():
            a,d=self.ui_q.get()
            if   a=="progress": self.ui.progressBar.setValue(int(d))
            elif a=="log":      self.ui.logTextEdit.append(d)
            elif a=="loadScan": self._load_vol(d)
            elif a=="segmentation": self._load_seg(d)

    def _browse(self,cap,le,attr):
        p=qt.QFileDialog.getExistingDirectory(self.parent,cap)
        if p: setattr(self,attr,p); getattr(self.ui,le).setText(p)

    # ───────── model download ────────────────────────────────────────────────
    def _download_model(self):
        import requests
        url="https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/download/CLIC_model/final_model.pth"
        dst_dir=Path.home()/ "Documents"/"CLIC_Models"; dst_dir.mkdir(exist_ok=True)
        dst=dst_dir/"final_model.pth"
        try:
            self.ui.PredScanLabel.setText("Downloading …")
            self.ui.PredScanProgressBar.setVisible(True)
            r=requests.get(url,stream=True,timeout=120); r.raise_for_status()
            tot=int(r.headers.get("content-length",0)); done=0
            with open(dst,"wb")as f:
                for chunk in r.iter_content(1<<20):
                    f.write(chunk); done+=len(chunk)
                    if tot: self.ui.PredScanProgressBar.setValue(done*100//tot)
            self.model_dir=str(dst_dir); self.ui.lineEditModelPath.setText(str(dst_dir))
            self.ui.PredScanLabel.setText("Model ✔")
        except Exception as e:
            qt.QMessageBox.warning(self.parent,"Download",str(e))
            self.ui.PredScanLabel.setText("Download failed.")
        finally:
            self.ui.PredScanProgressBar.setVisible(False)

    # ───────── volume / seg & legend ─────────────────────────────────────────
    def _load_vol(self,p): v=slicer.util.loadVolume(p); slicer.util.setSliceViewerLayers(background=v)
    def _load_seg(self,p):
        s=slicer.util.loadSegmentation(p); self.currentSegNode=s; self._legend()

    def _legend(self):
        import vtk
        n=self.currentSegNode
        if not n: return
        names={1:"Buccal",2:"Bicortical",3:"Palatal"};cols={1:(0,1,0),2:(1,1,0),3:(0.6,0.4,0.2)}
        seg=n.GetSegmentation(); ids=vtk.vtkStringArray(); seg.GetSegmentIDs(ids)
        for i in range(ids.GetNumberOfValues()):
            seg.GetSegment(ids.GetValue(i)).SetColor(*cols.get(i+1,(1,1,1)))
        lm=slicer.app.layoutManager()
        for vn in ("Red","Yellow","Green"):
            try:
                view=lm.sliceWidget(vn).sliceView(); ren=view.renderWindow().GetRenderers().GetFirstRenderer()
                for a in list(ren.GetActors2D()):
                    if getattr(a,"_leg",False): ren.RemoveActor(a)
                y0,dy,fs=0.85,0.06,16
                for k,l in names.items():
                    t=vtk.vtkTextActor(); t._leg=True; t.SetInput("■ "+l)
                    tp=t.GetTextProperty(); tp.SetFontSize(fs); tp.SetColor(*cols[k]); tp.BoldOn()
                    t.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay(); t.SetPosition(0.77,y0-(k-1)*dy)
                    ren.AddActor2D(t)
                view.forceRender()
            except Exception: pass

    def _on_sh_modified(self,caller,event):
        sh=caller; nid=sh.GetActiveItemID()
        if nid:
            n=sh.GetItemDataNode(nid)
            if n and n.IsA("vtkMRMLSegmentationNode"):
                self.currentSegNode=n; self._legend()

    def initializeParameterNode(self): pass
