#!/usr/bin/env python-real

import argparse
import vtk
from Method.make_butterfly import butterflyPatch
from Method.draw import drawPatch
from Method.ICP import vtkICP,ICP
from Method.vtkSegTeeth import vtkMeshTeeth
import os 
import numpy as np


def main(args):
    # Read the file (coordinate using : LPS)
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(args.lineedit)
    reader.Update()
    modelNode = reader.GetOutput()

    # Transform the data to read it in coordinate RAS (like slicer)
    transform = vtk.vtkTransform()
    transform.Scale(-1, -1, 1)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(modelNode)
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    modelNode = transformFilter.GetOutput()
   

    if args.type=="butterfly":
        butterflyPatch(modelNode,
                        args.lineedit_teeth_left_top,
                        args.lineedit_teeth_right_top,
                        args.lineedit_teeth_left_bot,
                        args.lineedit_teeth_right_bot,
                        args.lineedit_ratio_left_top,
                        args.lineedit_ratio_right_top,
                        args.lineedit_ratio_left_bot,
                        args.lineedit_ratio_right_bot,
                        args.lineedit_adjust_left_top,
                        args.lineedit_adjust_right_top,
                        args.lineedit_adjust_left_bot,
                        args.lineedit_adjust_right_bot)
    
    elif args.type=="curve":
        # Reading the data
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

        drawPatch(curve,modelNode,middle)

    elif args.type=="icp":
        # Reading the T1 model to register
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(args.path_reg)
        reader.Update()
        modelNodeT1 = reader.GetOutput()

        # Transform the data to read it in coordinate RAS (like slicer)
        transform = vtk.vtkTransform()
        transform.Scale(-1, -1, 1)

        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputData(modelNodeT1)
        transformFilter.SetTransform(transform)
        transformFilter.Update()

        modelNodeT1 = transformFilter.GetOutput()

        # ICP
        methode = [vtkICP()]
        option = vtkMeshTeeth(list_teeth=[1], property="Butterfly")
        icp = ICP(methode, option=option)
        output_icp = icp.run(modelNode, modelNodeT1)

        matrix_array=output_icp["matrix"]

        vtk_matrix = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk_matrix.SetElement(i, j, matrix_array[i, j])

        # Apply the matrix to register
        transform = vtk.vtkTransform()
        transform.SetMatrix(vtk_matrix)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputData(modelNode)
        transformFilter.SetTransform(transform)
        transformFilter.Update()

        modelNode = transformFilter.GetOutput()
        modelNode.Modified()

        

    # Save the changement in modelNode
    modelNode.Modified()

    # Put back the data in the LPS coordinate
    inverseTransform = vtk.vtkTransform()
    inverseTransform.Scale(-1, -1, 1)

    inverseTransformFilter = vtk.vtkTransformPolyDataFilter()
    inverseTransformFilter.SetInputData(modelNode)
    inverseTransformFilter.SetTransform(inverseTransform)
    inverseTransformFilter.Update()

    modelNode = inverseTransformFilter.GetOutput()

    modelNode.Modified()

    # Save the new file with the model

    
    writer = vtk.vtkPolyDataWriter()
    if args.type!="icp":
        writer.SetFileName(args.lineedit)
    else:
        outpath = args.lineedit.replace(os.path.dirname(args.lineedit),args.path_output)
        if not os.path.exists(os.path.dirname(outpath)):
            os.makedirs(os.path.dirname(outpath))

        writer.SetFileName(outpath.split('.vtk')[0].split('vtp')[0]+args.suffix+'.vtk')

    writer.SetInputData(modelNode)
    writer.Write()

    print("dans cli apres traitement")

    







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
    
    


    args = parser.parse_args()


    main(args)