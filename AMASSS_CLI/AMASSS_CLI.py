#!/usr/bin/env python3
"""
AMASSS_CLI.py – Adaptation pour nnUNet v2 (MAX, MAND, CB)
"""

import time, os, sys, glob, subprocess, shutil
import numpy as np
import torch, itk, cc3d, dicom2nifti
import SimpleITK as sitk
import vtk
import slicer
import re
import vtk, qt, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin, pip_install
# ── Constantes globales ───────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSLATE = {
  "Mandible":"MAND","Maxilla":"MAX","Cranial-base":"CB",
  "Cervical-vertebra":"CV","Root-canal":"RC","Mandibular-canal":"MCAN",
  "Upper-airway":"UAW","Skin":"SKIN","Teeth":"TEETH",
  "Cranial Base (Mask)":"CBMASK","Mandible (Mask)":"MANDMASK","Maxilla (Mask)":"MAXMASK",
}
NTRANSLATE = {v:v for v in TRANSLATE.values()}

LABELS = {
    "LARGE":{"MAND":1,"CB":2,"UAW":3,"MAX":4,"CV":5,"SKIN":6,"CBMASK":7,"MANDMASK":8,"MAXMASK":9},
    "SMALL":{"MAND":1,"RC":2,"MAX":4},
}
LABEL_COLORS = {1:[216,101,79],2:[128,174,128],3:[0,0,0],4:[230,220,70],5:[111,184,210],6:[172,122,101]}
NAMES_FROM_LABELS = {"LARGE":{}, "SMALL":{}}
for g,d in LABELS.items():
    for k,v in d.items():
        NAMES_FROM_LABELS[g][v] = k

MODELS_GROUP = {
    "LARGE":{
        "FF":     {"MAND":1,"CB":2,"UAW":3,"MAX":4,"CV":5},
        "SKIN":   {"SKIN":1},
        "CBMASK": {"CBMASK":1},
        "MANDMASK":{"MANDMASK":1},
        "MAXMASK":{"MAXMASK":1},
    },
    "SMALL":{
        "HD-MAND":{"MAND":1},
        "HD-MAX": {"MAX":1},
        "RC":     {"RC":1},
    },
}

# ── Fonctions utilitaires (inchangées) ──────────────────────────────────────
def CorrectHisto(filepath,outpath,min_porcent=0.01,max_porcent=0.95,i_min=-1500,i_max=4000):
    print("Correcting scan contrast:", filepath)
    img = sitk.Cast(sitk.ReadImage(filepath), sitk.sitkFloat32)
    # arr = sitk.GetArrayFromImage(img)
    # mn, mx = arr.min(), arr.max()
    # definition=1000
    # histo,bin_edges = np.histogram(arr,definition)
    # cum = np.cumsum(histo); cum=(cum-cum.min())/cum.max()
    # ih = np.where(cum>max_porcent)[0][0]; rh=bin_edges[ih]
    # il = np.where(cum>min_porcent)[0][0]; rl=bin_edges[il]
    # rl, rh = max(rl,i_min), min(rh,i_max)
    # arr = np.clip(arr, rl, rh)
    # out = sitk.GetImageFromArray(arr); out.CopyInformation(img)
    # out = sitk.Cast(out, sitk.sitkInt16)
    # sitk.WriteImage(out, outpath)
    return img



def Write(vtkdata, output_name):
	outfilename = output_name
	print("Writting:", outfilename)
	polydatawriter = vtk.vtkPolyDataWriter()
	polydatawriter.SetFileName(outfilename)
	polydatawriter.SetInputData(vtkdata)
	polydatawriter.Write()




import os
import numpy as np
import SimpleITK as sitk
import vtk

# (Supposer que LABEL_COLORS et Write(model, outpath) sont déjà définis globalement)

import os, numpy as np, SimpleITK as sitk, vtk
from vtk.util.numpy_support import vtk_to_numpy

def SavePredToVTK(file_path, temp_folder, smoothing, vtk_output_path, model_size="LARGE"):
    import os, numpy as np, SimpleITK as sitk, vtk

    # 1) Lecture
    img = sitk.ReadImage(file_path)
    arr = sitk.GetArrayFromImage(img)

    # 2) Prépare le nom de base (sans extensions multiples)
    base = os.path.basename(file_path)
    for ext in ('.nii.gz','.nrrd.gz','.nii','.nrrd'):
        if base.endswith(ext):
            base = base[:-len(ext)]
            break

    # 3) On détecte le mode merged
    is_merged = base.endswith("_MERGED")

    # 4) Crée le dossier de sortie si besoin
    output_is_dir = vtk_output_path.endswith(os.sep) or os.path.isdir(vtk_output_path)
    if output_is_dir:
        os.makedirs(vtk_output_path, exist_ok=True)

    # --- UTIL : écrivain VTK ---
    def write_poly(poly, outvtk):
        w = vtk.vtkPolyDataWriter()
        w.SetFileName(outvtk)
        w.SetInputData(poly)
        w.Write()
        print(f" → Written VTK: {outvtk}", flush=True)

    # --- UTIL : build + smooth + color ---
    def mesh_from_nrrd(nrrd, iters, color_rgb):
        r = vtk.vtkNrrdReader()
        r.SetFileName(nrrd)
        r.Update()
        dmc = vtk.vtkDiscreteMarchingCubes()
        dmc.SetInputConnection(r.GetOutputPort())
        dmc.GenerateValues(1, 1, 1)  # binaire
        s = vtk.vtkSmoothPolyDataFilter()
        s.SetInputConnection(dmc.GetOutputPort())
        s.SetNumberOfIterations(iters)
        s.Update()
        poly = s.GetOutput()
        cols = vtk.vtkUnsignedCharArray()
        cols.SetName("Colors")
        cols.SetNumberOfComponents(3)
        cols.SetNumberOfTuples(poly.GetNumberOfCells())
        for i in range(poly.GetNumberOfCells()):
            cols.SetTuple(i, color_rgb)
        poly.GetCellData().SetScalars(cols)
        return poly

    # ——— MODE MERGED ———
    if is_merged:
        # 1) On va append tous les sous-maillages colorés label par label
        append = vtk.vtkAppendPolyData()
        for label in sorted(np.unique(arr)):
            if label == 0:
                continue
            struct = NAMES_FROM_LABELS[model_size][label]
            # écriture NRRD temporaire pour ce label
            tmp_nrrd = os.path.join(temp_folder, f"temp.nrrd")
            mask = (arr == label).astype(np.uint8)
            img2 = sitk.GetImageFromArray(mask); img2.CopyInformation(img)
            sitk.WriteImage(img2, tmp_nrrd)
            # extraction + lissage + couleur
            color = LABEL_COLORS.get(label, [255,255,255])
            mesh = mesh_from_nrrd(tmp_nrrd, smoothing, color)
            append.AddInputData(mesh)
        append.Update()
        merged_poly = append.GetOutput()

        # 2) Écriture du polydata final
        outname = f"{base}.vtk"
        if output_is_dir:
            outvtk = os.path.join(vtk_output_path, outname)
        else:
            root, _ = os.path.splitext(vtk_output_path)
            outvtk = f"{root}_{outname}"
            os.makedirs(os.path.dirname(outvtk), exist_ok=True)
        write_poly(merged_poly, outvtk)
        return

    # ——— MODE SEPARATE ———
    # On déduit le nom de la structure depuis le suffixe du fichier
    struct = base.split('_')[-1]  # ex: "MAND", "MAX", etc.
    # écriture du NRRD binaire
    tmp_nrrd = os.path.join(temp_folder, f"temp.nrrd")
    m = (arr > 0).astype(np.uint8)
    i2 = sitk.GetImageFromArray(m); i2.CopyInformation(img)
    sitk.WriteImage(i2, tmp_nrrd)
    # récupération de l'index et de la couleur
    label_index = LABELS[model_size][struct]
    color = LABEL_COLORS.get(label_index, [255,255,255])
    # build + write
    poly = mesh_from_nrrd(tmp_nrrd, smoothing, color)
    outname = f"{base}.vtk"
    outvtk = os.path.join(vtk_output_path, outname) if output_is_dir else os.path.join(os.path.dirname(vtk_output_path), outname)
    write_poly(poly, outvtk)





def SetSpacingFromRef(filepath,refFile,interpolator="NearestNeighbor",outpath=-1):
    img = itk.imread(filepath); ref = itk.imread(refFile)
    sp_i, sz_i = np.array(img.GetSpacing()), np.array(itk.size(img))
    sp_r, sz_r = np.array(ref.GetSpacing()), np.array(itk.size(ref))
    if not np.allclose(sp_i,sp_r) or not np.array_equal(sz_i,sz_r):
        PixelType = itk.template(img)[1][0]
        Dim=3
        IVec = itk.Image[PixelType,Dim]
        interp = itk.NearestNeighborInterpolateImageFunction[IVec, itk.D].New() \
                 if interpolator=="NearestNeighbor" else itk.LinearInterpolateImageFunction[IVec,itk.D].New()
        res = itk.ResampleImageFilter[IVec,IVec].New(Input=img,
             OutputSpacing=sp_r.tolist(), OutputOrigin=ref.GetOrigin(),
             OutputDirection=ref.GetDirection(), Interpolator=interp,
             Size=sz_r.tolist()); res.Update()
        out = sitk.GetImageFromArray(itk.GetArrayFromImage(res.GetOutput()))
        out.CopyInformation(sitk.ReadImage(refFile))
    else:
        out = sitk.ReadImage(filepath)
    out = sitk.Cast(out, sitk.sitkInt16)
    if outpath!=-1: sitk.WriteImage(out, outpath)
    return out

def CleanArray(seg_arr,radius):
    img = sitk.GetImageFromArray(seg_arr.astype(np.uint8))
    img = sitk.BinaryDilate(img,[radius]*3)
    img = sitk.BinaryFillhole(img)
    img = sitk.BinaryErode(img,[radius]*3)
    arr = sitk.GetArrayFromImage(img)
    cc, n = cc3d.connected_components(arr, return_N=True)
    if n>1:
        sizes = [(cc==i).sum() for i in range(1,n+1)]
        arr = (cc== (1+int(np.argmax(sizes)))).astype(np.uint8)
    return arr

def CropSkin(skin_seg_arr,thickness):
    img = sitk.GetImageFromArray(skin_seg_arr.astype(np.uint8))
    fill = sitk.BinaryFillhole(img)
    ero = sitk.BinaryErode(fill,[thickness]*3)
    arr = sitk.GetArrayFromImage(fill)
    earr = sitk.GetArrayFromImage(ero)
    crop = np.where(earr==1,0,arr)
    cc, n = cc3d.connected_components(crop,return_N=True)
    if n>1:
        sizes=[(cc==i).sum() for i in range(1,n+1)]
        crop=(cc==(1+int(np.argmax(sizes)))).astype(np.uint8)
    return crop

def SavePrediction(img,ref_filepath,outpath,output_spacing):
    ref = sitk.ReadImage(ref_filepath)
    out = sitk.GetImageFromArray(img.astype(np.int16))
    out.SetSpacing(output_spacing)
    out.SetDirection(ref.GetDirection())
    out.SetOrigin(ref.GetOrigin())
    sitk.WriteImage(out, outpath)

def SaveSeg(file_path, spacing ,seg_arr, input_path,temp_path, outputdir,temp_folder, save_vtk, smoothing = 5, model_size= "LARGE"):

    print("Saving segmentation for ", file_path)

    SavePrediction(seg_arr,input_path,temp_path,output_spacing = spacing)
    # if clean_seg:
    #     CleanScan(temp_path)
    SetSpacingFromRef(
        temp_path,
        input_path,
        # "Linear",
        outpath=file_path
        )

    if save_vtk:
        SavePredToVTK(file_path,temp_folder, smoothing, vtk_output_path=outputdir)
# ── Main adapté nnUNet v2 ─────────────────────────────────────────────────────
def main(args):
    import os, sys, glob, subprocess, shutil, time
    import numpy as np
    import torch
    import SimpleITK as sitk

    print("Start AMASSS_CLI with nnUNet v2 backend", flush=True)

    # 1. Préparation du dossier temporaire
    tmp = args["temp_fold"]
    base_output = args["output_folder"]
    os.makedirs(tmp, exist_ok=True)

    # 2. Construction de la liste des scans à traiter
    input_path = args["inputVolume"]
    extensions = (".nii", ".nii.gz", ".nrrd", ".nrrd.gz")
    if os.path.isdir(input_path):
        input_files = sorted([
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(extensions)
        ])
    else:
        input_files = [input_path]
    scan_count = len(input_files)
    print(f"Nombre de scans à traiter : {scan_count}", flush=True)

    # 3. Début du filtrage Slicer
    start_time = time.time()
    print("<filter-start><filter-name>AMASSS</filter-name></filter-start>", flush=True)
    sys.stdout.flush()

    # 4. Boucle sur chaque scan
    for scan_idx, volume_file in enumerate(input_files, start=1):
        case_id = f"{scan_idx:03d}"
        basename = os.path.basename(volume_file)
        base, ext = os.path.splitext(basename)
        if ext == ".gz":
            base, ext2 = os.path.splitext(base)
            ext = ext2 + ext

        # --- Choix du dossier de sortie (grouped ou pas) ---
        if args.get("save_in_folder"):
            outdir = os.path.join(base_output, f"{base}_{args['prediction_ID']}_SegOut")
        else:
            outdir = base_output
        os.makedirs(outdir, exist_ok=True)

        print(f"\n--- Traitement du scan {scan_idx}/{scan_count} : {basename} ---", flush=True)

        # 4.1 Copier et renommer pour nnUNet
        tmp_name = f"p_{case_id}_0000.nii.gz"
        input_vol = os.path.join(tmp, tmp_name)
        shutil.copy(volume_file, input_vol)
        print(f"→ Copied and renamed to {input_vol}", flush=True)

        # 4.2 Recherche des modèles nnUNet
        nnunet_models = {}
        for struct in args["skullStructure"].split(","):
            root = os.path.join(args["modelDirectory"], struct)
            pattern = os.path.join(root, "**", "*__nnUNetPlans__3d_fullres")
            plans = glob.glob(pattern, recursive=True)
            if plans:
                nnunet_models[struct] = plans[0]
            print(f"  {struct}: {plans}", flush=True)
        if not nnunet_models:
            sys.exit("❌ Aucun modèle nnUNet v2 trouvé dans votre arborescence.")

        total_struct = len(nnunet_models)
        total_steps = scan_count * total_struct
        prediction_segmentation = {}

        # 4.3 Prédiction et post‑traitement
        for struct_idx, (struct, plans_dir) in enumerate(nnunet_models.items(), start=1):
            dataset_name = os.path.basename(os.path.dirname(plans_dir))
            os.environ['nnUNet_results'] = os.path.dirname(os.path.dirname(plans_dir))
            outp = os.path.join(tmp, f"pred_{struct}")
            os.makedirs(outp, exist_ok=True)

            checkpoint = os.path.join(plans_dir, "fold_0", "checkpoint_final.pth")
            if not os.path.isfile(checkpoint):
                sys.exit(f"❌ Pas de checkpoint trouvé pour {struct} à {checkpoint}")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  → Predicting {struct} on {device}", flush=True)
            cmd = [
                "nnUNetv2_predict",
                "-i", tmp,
                "-o", outp,
                "-d", dataset_name,
                "-c", "3d_fullres",
                "-f", "0",
                "-device", device,
                "--disable_tta"
            ]
            subprocess.check_call(cmd)

            # Émission de la progression APRÈS la segmentation
            step = (scan_idx - 1) * total_struct + struct_idx
            fraction = step / total_steps
            print(f"<filter-progress>{fraction:.4f}</filter-progress>", flush=True)
            sys.stdout.flush()

            # Lecture et stockage du masque
            nifti_pred = os.path.join(outp, f"p_{case_id}.nii.gz")
            if not os.path.isfile(nifti_pred):
                sys.exit(f"❌ Prédiction introuvable : {nifti_pred}")
            img = sitk.ReadImage(nifti_pred)
            arr = sitk.GetArrayFromImage(img)
            mask = (arr > 0).astype(np.uint8)
            prediction_segmentation[struct] = mask

        # 4.4 Sauvegarde des segmentations
        spacing = list(sitk.ReadImage(volume_file).GetSpacing())

        # Cas séparé
        if "SEPARATE" in args["merge"] or len(prediction_segmentation) == 1:
            for struct, mask in prediction_segmentation.items():
                outfn = os.path.join(outdir, f"{base}_{args['prediction_ID']}_{struct}{ext}")
                SaveSeg(
                    outfn, spacing, mask, volume_file,
                    os.path.join(tmp, "tmp.nii.gz"),
                    outdir, tmp, args["genVtk"], args["vtk_smooth"], "LARGE"
                )

        # Cas fusionné
        if "MERGE" in args["merge"] and len(prediction_segmentation) > 1:
            shape = next(iter(prediction_segmentation.values())).shape
            merged = np.zeros(shape, dtype=np.int16)
            for struct in args["merging_order"]:
                if struct in prediction_segmentation:
                    lbl = LABELS["LARGE"].get(struct, 1)
                    merged = np.where(prediction_segmentation[struct] == 1, lbl, merged)
            outfn = os.path.join(outdir, f"{base}_{args['prediction_ID']}_MERGED{ext}")
            SaveSeg(
                outfn, spacing, merged, volume_file,
                os.path.join(tmp, "tmp.nii.gz"),
                outdir, tmp, args["genVtk"], args["vtk_smooth"], "LARGE"
            )

        # Nettoyage temporaire pour ce scan
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)

    # 5. Fin du filtrage
    elapsed = time.time() - start_time
    print(f"<filter-end><filter-name>AMASSS</filter-name><filter-time>{elapsed:.2f}</filter-time></filter-end>", flush=True)
    sys.stdout.flush()
    print("Done.", flush=True)


# ── Entrée CLI ────────────────────────────────────────────────────────────────
if __name__=="__main__":
    # Exemple d’appel :
    # python AMASSS_CLI.py \
    #   /path/to/input.nii.gz \
    #   /path/to/AMASSS_Models \
    #   false \
    #   MAND,MAX,CB \
    #   MERGE,SEPARATE \
    #   false \
    #   false \
    #   /output/dir \
    #   50 \
    #   5 \
    #   Pred \
    #   1 \
    #   false \
    #   /tmp/slicer_amasss \
    #   false \
    #   false

    argv = sys.argv
    args = {
        "inputVolume":    argv[1],
        "modelDirectory": argv[2],
        "highDefinition": argv[3].lower()=="true",
        "skullStructure": argv[4],
        "merge":          re.split(r'[, ]+', argv[5].strip()),  
        "genVtk":         argv[6].lower()=="true",
        "save_in_folder": argv[7].lower()=="true",
        "output_folder":  argv[8],
        "vtk_smooth":     int(argv[10]),
        "prediction_ID":  argv[11],
        "temp_fold":      argv[14],
        "isSegmentInput": argv[15].lower()=="true",
        "isDCMInput":     argv[16].lower()=="true",
        "merging_order":  ["SKIN","CV","UAW","CB","MAX","MAND","CAN","RC","CBMASK","MANDMASK","MAXMASK"],
    }
    main(args)