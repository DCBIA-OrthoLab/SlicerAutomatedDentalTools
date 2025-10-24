#!/usr/bin/env python3
"""
AMASSS_CLI.py – Adaptation pour nnUNet v2 (MAX, MAND, CB)
"""
import argparse
import time, os, sys, glob, subprocess, shutil
import numpy as np
import torch, itk, cc3d, dicom2nifti
import SimpleITK as sitk
import vtk
# import slicer
import re
import vtk
# qt, slicer
# from slicer.ScriptedLoadableModule import *
# from slicer.util import VTKObservationMixin, pip_install

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



# (suppose  LABEL_COLORS and Write(model, outpath) are global variable)

def SavePredToVTK(file_path, temp_folder, smoothing, vtk_output_path, model_size="LARGE"):
    import os, numpy as np, SimpleITK as sitk, vtk

    img = sitk.ReadImage(file_path)
    arr = sitk.GetArrayFromImage(img)

    base = os.path.basename(file_path)
    for ext in ('.nii.gz','.nrrd.gz','.nii','.nrrd'):
        if base.endswith(ext):
            base = base[:-len(ext)]
            break

    is_merged = base.endswith("_MERGED")

    output_is_dir = vtk_output_path.endswith(os.sep) or os.path.isdir(vtk_output_path)
    if output_is_dir:
        os.makedirs(vtk_output_path, exist_ok=True)

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
        # merge all label maps in one map
        append = vtk.vtkAppendPolyData()
        for label in sorted(np.unique(arr)):
            if label == 0:
                continue
            struct = NAMES_FROM_LABELS[model_size][label]

            tmp_nrrd = os.path.join(temp_folder, f"temp.nrrd")
            mask = (arr == label).astype(np.uint8)
            img2 = sitk.GetImageFromArray(mask); img2.CopyInformation(img)
            sitk.WriteImage(img2, tmp_nrrd)
            color = LABEL_COLORS.get(label, [255,255,255])
            mesh = mesh_from_nrrd(tmp_nrrd, smoothing, color)
            append.AddInputData(mesh)
        append.Update()
        merged_poly = append.GetOutput()

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
    # save each label map independently
    struct = base.split('_')[-1]  # ex: "MAND", "MAX", etc.

    tmp_nrrd = os.path.join(temp_folder, f"temp.nrrd")
    m = (arr > 0).astype(np.uint8)
    i2 = sitk.GetImageFromArray(m); i2.CopyInformation(img)
    sitk.WriteImage(i2, tmp_nrrd)

    label_index = LABELS[model_size][struct]
    color = LABEL_COLORS.get(label_index, [255,255,255])

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

    # args['merge']=re.split(r'[, ]+', args['merge'].strip())
    # args['genVtk']=args['genVtk'].lower()=="true"
    # args['save_in_folder']=args['save_in_folder'].lower()=="true"
    # args['isSegmentInput']=args['isSegmentInput'].lower=='true'
    # args['isDCMInput']=args['isDCMInput'].lower()=='true'


    tmp = args["temp_fold"]
    base_output = args["output_folder"]
    os.makedirs(tmp, exist_ok=True)

    input_path = args["inputVolume"]
    extensions = (".nii", ".nii.gz", ".nrrd", ".nrrd.gz")
    if os.path.isdir(input_path):
        input_files = []
        for f in os.listdir(input_path):
            file = os.path.join(input_path, f)
            if f.lower().endswith(extensions) :
                print(f)
                if ('MASK' not in f):
                    print('nope')
                    input_files.append(file)
    else:
        input_files = [input_path]
    scan_count = len(input_files)
    print(f"Number of scans : {scan_count}", flush=True)

    start_time = time.time()
    print("<filter-start><filter-name>AMASSS</filter-name></filter-start>", flush=True)
    sys.stdout.flush()

    for scan_idx, volume_file in enumerate(input_files, start=1):
        case_id = f"{scan_idx:03d}"
        basename = os.path.basename(volume_file)
        base, ext = os.path.splitext(basename)
        if ext == ".gz":
            base, ext2 = os.path.splitext(base)
            ext = ext2 + ext

        # --- choice of output folder ---
        if args.get("save_in_folder"):
            outdir = os.path.join(base_output, f"{base}_{args['prediction_ID']}_SegOut")
        else:
            outdir = base_output
        os.makedirs(outdir, exist_ok=True)

        print(f"\n--- Processing scan {scan_idx}/{scan_count} : {basename} ---", flush=True)

        tmp_name = f"p_{case_id}_0000.nii.gz"
        input_vol = os.path.join(tmp, tmp_name)
        shutil.copy(volume_file, input_vol)
        print(f"    Copied and renamed to {input_vol} for nnUnet", flush=True)

        # 4.2 Searching NNunet
        nnunet_models = {}
        for struct in args["skullStructure"].split(","):
            root = os.path.join(args["modelDirectory"], struct)
            pattern = os.path.join(root, "**", "*__nnUNetPlans__3d_fullres")
            plans = glob.glob(pattern, recursive=True)
            if plans:
                nnunet_models[struct] = plans[0]
            print(f"  {struct}: {plans}", flush=True)
        if not nnunet_models:
            sys.exit("❌ No model found.")

        total_struct = len(nnunet_models)
        total_steps = scan_count * total_struct
        prediction_segmentation = {}

        # 4.3 Predicting and post-processing
        for struct_idx, (struct, plans_dir) in enumerate(nnunet_models.items(), start=1):
            dataset_name = os.path.basename(os.path.dirname(plans_dir))
            os.environ['nnUNet_results'] = os.path.dirname(os.path.dirname(plans_dir))
            outp = os.path.join(tmp, f"pred_{struct}")
            os.makedirs(outp, exist_ok=True)

            checkpoint = os.path.join(plans_dir, "fold_0", "checkpoint_final.pth")
            if not os.path.isfile(checkpoint):
                sys.exit(f"❌ No model checkpoint found for {struct} in {checkpoint}")

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

            step = (scan_idx - 1) * total_struct + struct_idx
            fraction = step / total_steps
            print(f"<filter-progress>{fraction:.4f}</filter-progress>", flush=True)
            sys.stdout.flush()

            nifti_pred = os.path.join(outp, f"p_{case_id}.nii.gz")
            if not os.path.isfile(nifti_pred):
                sys.exit(f"❌ File not found : {nifti_pred}")
            img = sitk.ReadImage(nifti_pred)
            arr = sitk.GetArrayFromImage(img)
            mask = (arr > 0).astype(np.uint8)
            prediction_segmentation[struct] = mask

        spacing = list(sitk.ReadImage(volume_file).GetSpacing())

        if "SEPARATE" in args["merge"] or len(prediction_segmentation) == 1:
            for struct, mask in prediction_segmentation.items():
                outfn = os.path.join(outdir, f"{base}_{args['prediction_ID']}_{struct}{ext}")
                SaveSeg(
                    outfn, spacing, mask, volume_file,
                    os.path.join(tmp, "tmp.nii.gz"),
                    outdir, tmp, args["genVtk"], args["vtk_smooth"], "LARGE"
                )

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

        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)

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
    print(sys.argv)
    args = {
        "inputVolume":    argv[1],
        "modelDirectory": argv[2],
        "skullStructure": argv[3],
        "merge":          re.split(r'[, ]+', argv[4].strip()),  
        "genVtk":         argv[5].lower()=="true",
        "save_in_folder": argv[6].lower()=="true",
        "output_folder":  argv[7],
        "vtk_smooth":     int(argv[8]),
        "prediction_ID":  argv[9],        
        "temp_fold":      argv[10],
        "isSegmentInput": argv[11].lower()=="true",
        "isDCMInput":     argv[12].lower()=="true",
        "merging_order":  ["SKIN","CV","UAW","CB","MAX","MAND","CAN","RC","CBMASK","MANDMASK","MAXMASK"],
    }
    print(args)
    main(args)