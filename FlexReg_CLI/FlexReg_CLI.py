#!/usr/bin/env python-real

import argparse
import vtk
import SimpleITK as sitk
import os
import sys
# ADTLib sits next to the modules in an installed build, in the directory Slicer
# already has on sys.path. A source tree has no such entry -- a module search
# path only gets there once Slicer finds a module in it, and ADT holds none --
# so the entry points walk up to the holder directory and add it themselves.
_adt_root = os.path.dirname(os.path.realpath(__file__))
while not os.path.isdir(os.path.join(_adt_root, "ADT", "ADTLib")) \
        and _adt_root != os.path.dirname(_adt_root):
    _adt_root = os.path.dirname(_adt_root)
if os.path.join(_adt_root, "ADT") not in sys.path:
    sys.path.append(os.path.join(_adt_root, "ADT"))

from FlexReg_Method.make_butterfly import butterflyPatch
from FlexReg_Method.draw import drawPatch
from FlexReg_Method.ICP import vtkICP,ICP
from FlexReg_Method.vtkSegTeeth import vtkMeshTeeth
import os
import numpy as np
import torch
from vtk.util.numpy_support import vtk_to_numpy,numpy_to_vtk

import sys

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger

logger = get_logger("FlexReg_CLI")


def _register_with_icp(args, modelNode):
    """Register the mesh onto T1 by ICP, restricted to the patch.

    The matrix is also written as a `.tfm`, in the frame of the file: hence
    the composition with the flip applied to the input."""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(args.path_reg)
    reader.Update()
    model_node_t1 = reader.GetOutput()

    # Transform the data to read it in coordinate RAS (like slicer)
    transform = vtk.vtkTransform()
    transform.Scale(-1, -1, 1)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(model_node_t1)
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    model_node_t1 = transform_filter.GetOutput()

    if args.lower_arch != "None":
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(args.lower_arch)
        reader.Update()
        model_node_lower_arch = reader.GetOutput()

        transform = vtk.vtkTransform()
        transform.Scale(-1, -1, 1)

        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputData(model_node_lower_arch)
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        model_node_lower_arch = transform_filter.GetOutput()

    # ICP
    methode = [vtkICP()]
    # The palate patch and the mucogingival band live in arrays of their own,
    # so the arch decides which one the registration is computed on.
    patch_array = "Bottom_MGL" if args.type == "icp_mgl" else "Butterfly"
    logger.info(f"registering on the {patch_array} patch")
    option = vtkMeshTeeth(list_teeth=[1], property=patch_array)
    icp = ICP(methode, option=option)
    output_icp = icp.run(modelNode, model_node_t1)

    matrix_array=output_icp["matrix"]
    logger.info(f"matrix output icp : {matrix_array}")

    vtk_matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_matrix.SetElement(i, j, matrix_array[i, j])

    # Apply the matrix to register
    transform = vtk.vtkTransform()
    transform.SetMatrix(vtk_matrix)
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(modelNode)
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    # Save the registration matrix
    flip = np.diag([-1, -1, 1, 1])
    composed = flip @ matrix_array @ flip
    composed_inv = np.linalg.inv(composed)

    sitk_tfm = sitk.AffineTransform(3)
    sitk_tfm.SetMatrix(composed_inv[:3, :3].flatten())
    sitk_tfm.SetTranslation(composed_inv[:3, 3])

    base_filename = os.path.splitext(os.path.basename(args.lineedit))[0]
    tfm_outpath = os.path.join(args.path_output, f"{base_filename}{args.suffix}.tfm")
    sitk.WriteTransform(sitk_tfm, tfm_outpath)
    logger.info(f"Saved inverted matrix to: {tfm_outpath}")

    modelNode = transform_filter.GetOutput()
    modelNode.Modified()

    if args.lower_arch != "None":
        transform = vtk.vtkTransform()
        transform.SetMatrix(vtk_matrix)
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputData(model_node_lower_arch)
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        model_node_lower_arch = transform_filter.GetOutput()
        model_node_lower_arch.Modified()
        model_node_lower_arch.Modified()
    return modelNode, model_node_lower_arch

def _delete_patch(args, modelNode):
    """Remove one patch and renumber the ones that follow it."""
    index = args.index_patch + 1
    while True:
        array_name = f"Butterfly{index}"

        # Check if array existing
        if modelNode.GetPointData().HasArray(array_name):

            current_array = modelNode.GetPointData().GetArray(array_name)

            # Rename it
            new_array_name = f"Butterfly{index-1}"
            current_array.SetName(new_array_name)

            # Update old array by the new one
            modelNode.GetPointData().AddArray(current_array)

            index += 1
        else:
            break

    # Delete last array
    modelNode.GetPointData().RemoveArray(f"Butterfly{index-1}")

def _draw_curve_patch(args, modelNode):
    """Lay a patch along the curve drawn by the user."""
    vector_middle = args.middle_point[1:-1]
    x, y, z = map(float, vector_middle.split(','))
    middle = vtk.vtkVector3d(x, y, z)


    # Splitting the string into individual array-like strings
    array_strings = args.curve.split('],[')

    # Initializing an empty list to store the ndarrays
    arrays = []

    # Looping through each array-like string to convert them into numpy arrays
    for array_string in array_strings:
        # Removing the brackets and splitting by spaces to get individual numbers
        numbers = array_string.replace('[', '').replace(']', '').split()
        # Converting the numbers into a numpy array and appending to the list
        arrays.append(np.array([float(num) for num in numbers]))

    curve =[arr.astype(np.float32) for arr in arrays]

    drawPatch(curve,modelNode,middle,args.index_patch)

def _apply_butterfly_patch(args, modelNode):
    """Lay the butterfly patch from the three given points."""
    logger.info(
            f"Teeth: LT={args.lineedit_teeth_left_top}, RT={args.lineedit_teeth_right_top}, "
            f"LB={args.lineedit_teeth_left_bot}, RB={args.lineedit_teeth_right_bot} | "
            f"Ratios: LT={args.lineedit_ratio_left_top}, RT={args.lineedit_ratio_right_top}, "
            f"LB={args.lineedit_ratio_left_bot}, RB={args.lineedit_ratio_right_bot} | "
            f"Adjust: LT={args.lineedit_adjust_left_top}, RT={args.lineedit_adjust_right_top}, "
            f"LB={args.lineedit_adjust_left_bot}, RB={args.lineedit_adjust_right_bot} | "
            f"Shift: LR={args.shift_lr}, AP={args.shift_ap} | "
            f"Index: {args.index_patch}"
        )

    butterflyPatch(
        surf=modelNode,
        tooth_anterior_right=args.lineedit_teeth_right_top,
        tooth_anterior_left=args.lineedit_teeth_left_top,

        tooth_posterior_right=args.lineedit_teeth_right_bot,
        tooth_posterior_left=args.lineedit_teeth_left_bot,

        ratio_anterior_right=args.lineedit_ratio_right_top,
        ratio_anterior_left=args.lineedit_ratio_left_top,

        ratio_posterior_left=args.lineedit_ratio_left_bot,
        ratio_posterior_right=args.lineedit_ratio_right_bot,

        adjust_anterior_right=args.lineedit_adjust_right_top,
        adjust_anterior_left=args.lineedit_adjust_left_top,

        adjust_posterior_right=args.lineedit_adjust_right_bot,
        adjust_posterior_left=args.lineedit_adjust_left_bot,

        index=args.index_patch,

        shift_lr=args.shift_lr,
        shift_ap=args.shift_ap
    )

def main(args):
    logger.info(f"args.lower_arch : {args.lower_arch}")
    logger.info(f"index_patch :{args.index_patch}")
    # Read the file (coordinate using : LPS)
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(args.lineedit)
    reader.Update()
    model_node = reader.GetOutput()

    # Transform the data to read it in coordinate RAS (like slicer)
    transform = vtk.vtkTransform()
    transform.Scale(-1, -1, 1)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(model_node)
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    model_node = transform_filter.GetOutput()
   

    if args.type=="butterfly":

        _apply_butterfly_patch(args, model_node)
    
    elif args.type=="curve":
        # Reading the data
        _draw_curve_patch(args, model_node)

    elif args.type=="delete":
        # To delete the array it will rename all the array with a number > index and delete the last one
        _delete_patch(args, model_node)

    elif args.type in ("icp", "icp_mgl"):
        # Reading the T1 model to register
        model_node, model_node_lower_arch = _register_with_icp(args, model_node)
        
       

        

    # Save the changement in modelNode
    model_node.Modified()

    index = 1
    final_array = None

    # Create the patch Butterfly. Skipped for the lower arch: there is no
    # Butterfly1..N to merge there, and the pass needs a GPU the mucogingival
    # workflow does without.
    while args.type != "icp_mgl":
        array_name = f"Butterfly{index}"
        
        if model_node.GetPointData().HasArray(array_name):
            current_array = model_node.GetPointData().GetArray(array_name)
            current_tensor = torch.tensor(vtk_to_numpy(current_array)).to(torch.float32).cuda()
            
            if final_array is None:
                final_array = current_tensor
            else:
                final_array = torch.logical_or(final_array, current_tensor).to(torch.float32)
            
            index += 1
        else:
            break

    if args.type != "icp_mgl":
        if final_array is None:
            num_points = model_node.GetNumberOfPoints()
            v_label = torch.zeros(num_points).to(torch.float32).cuda()
        else:
            v_label = final_array

        v_labels_prediction = numpy_to_vtk(v_label.cpu().numpy())
        v_labels_prediction.SetName('Butterfly')
        model_node.GetPointData().AddArray(v_labels_prediction)


    # Put back the data in the LPS coordinate
    inverse_transform = vtk.vtkTransform()
    inverse_transform.Scale(-1, -1, 1)

    inverse_transform_filter = vtk.vtkTransformPolyDataFilter()
    inverse_transform_filter.SetInputData(model_node)
    inverse_transform_filter.SetTransform(inverse_transform)
    inverse_transform_filter.Update()

    model_node = inverse_transform_filter.GetOutput()

    model_node.Modified()

    # Save the new file with the model

    
    writer = vtk.vtkPolyDataWriter()
    # A registration writes its result to the output folder; the patch modes
    # write back into the scan they were given.
    if args.type not in ("icp", "icp_mgl"):
        writer.SetFileName(args.lineedit)
    else:
        outpath = args.lineedit.replace(os.path.dirname(args.lineedit),args.path_output)
        if not os.path.exists(os.path.dirname(outpath)):
            os.makedirs(os.path.dirname(outpath))

        writer.SetFileName(outpath.split('.vtk')[0].split('vtp')[0]+args.suffix+'.vtk')

    writer.SetInputData(model_node)
    writer.Write()
    
    if args.lower_arch != "None":
        # Put back the data in the LPS coordinate
        inverse_transform = vtk.vtkTransform()
        inverse_transform.Scale(-1, -1, 1)

        inverse_transform_filter = vtk.vtkTransformPolyDataFilter()
        inverse_transform_filter.SetInputData(model_node_lower_arch)
        inverse_transform_filter.SetTransform(inverse_transform)
        inverse_transform_filter.Update()

        model_node_lower_arch = inverse_transform_filter.GetOutput()

        model_node_lower_arch.Modified()
        
        outpath = args.lower_arch.replace(os.path.dirname(args.lower_arch),args.path_output)
        if not os.path.exists(os.path.dirname(outpath)):
            os.makedirs(os.path.dirname(outpath))

        writer.SetFileName(outpath.split('.vtk')[0].split('vtp')[0]+args.suffix+'.vtk')
        writer.SetInputData(model_node_lower_arch)
        writer.Write()


    







if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('lineedit',type=str)

    parser.add_argument('lineedit_teeth_left_top',type=int)
    parser.add_argument('lineedit_teeth_right_top',type=int)
    parser.add_argument('lineedit_teeth_left_bot',type=int)
    parser.add_argument('lineedit_teeth_right_bot',type=int)

    parser.add_argument('lineedit_ratio_left_top',type=float)
    parser.add_argument('lineedit_ratio_right_top',type=float)
    parser.add_argument('lineedit_ratio_left_bot',type=float)
    parser.add_argument('lineedit_ratio_right_bot',type=float)

    parser.add_argument('lineedit_adjust_left_top',type=float)
    parser.add_argument('lineedit_adjust_right_top',type=float)
    parser.add_argument('lineedit_adjust_left_bot',type=float)
    parser.add_argument('lineedit_adjust_right_bot',type=float)

    parser.add_argument('curve',type=str)
    parser.add_argument('middle_point',type=str)
    parser.add_argument('type',type=str)

    parser.add_argument('path_reg',type=str)
    parser.add_argument('path_output',type=str)
    parser.add_argument('suffix',type=str)

    parser.add_argument('index_patch',type=int)
    
    parser.add_argument('lower_arch',type=str)

    # butterfly : translation of the whole patch, in mm, in the oriented frame
    parser.add_argument('shift_lr',type=float)
    parser.add_argument('shift_ap',type=float)

    


    args = parser.parse_args()


    main(args)