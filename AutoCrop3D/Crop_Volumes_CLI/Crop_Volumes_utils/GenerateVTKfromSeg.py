import vtk
import SimpleITK as sitk
import numpy as np
import os

LABEL_COLORS = {
    1: [216, 101, 79],
    2: [128, 174, 128],
    3: [0, 0, 0],
    4: [230, 220, 70],
    5: [111, 184, 210],
    6: [172, 122, 101],
}

def convertNiftiToVTK(input_path, output_path ) -> None:
        """
        function to generate VTK files from nifti segmentation (labelmap).
        Input : path of your segmentation file, path of your futur vtk file
        """
        input_filename = input_path
        output_filename = output_path

        # Read nifti file and convert it to numpy array
        img = sitk.ReadImage(input_filename)
        img_arr = sitk.GetArrayFromImage(img)

        # Read all the labels present in the image
        present_labels = []
        for label in range(np.max(img_arr)):
                if label+1 in img_arr:
                        present_labels.append(label+1)

        label = np.max(img_arr)

        padded_image_filename = padding(img)

        # Read the original Nifti image
        surf = vtk.vtkNIFTIImageReader()
        surf.SetFileName(padded_image_filename)
        surf.Update()

        # Apply filter to create an isosurface from the input image
        # convert it to a 3D surfac representation
        dmc = vtk.vtkDiscreteMarchingCubes()
        dmc.SetInputConnection(surf.GetOutputPort())
        dmc.GenerateValues(100, 1, 100)

        # LAPLACIAN smooth to improve VTK rendering
        # by reducing the impact of jagged edges
        smooth_poly_data_filter = vtk.vtkSmoothPolyDataFilter()
        smooth_poly_data_filter.SetInputConnection(dmc.GetOutputPort())
        smooth_poly_data_filter.SetNumberOfIterations(5)
        smooth_poly_data_filter.SetFeatureAngle(120.0)
        smooth_poly_data_filter.SetRelaxationFactor(0.6)
        smooth_poly_data_filter.Update()

        model = smooth_poly_data_filter.GetOutput()

        # Coloring the VTK according to labels
        color = vtk.vtkUnsignedCharArray()
        color.SetName("Colors")
        color.SetNumberOfComponents(3)
        color.SetNumberOfTuples( model.GetNumberOfCells() )

        for i in range(model.GetNumberOfCells()):
            color_tup=LABEL_COLORS[label]
            color.SetTuple(i, color_tup)

        model.GetCellData().SetScalars(color)

        os.remove(padded_image_filename)
        # Save the final VTK model
        Write(model, output_filename)

def padding(input_image) :
    """
    Function to padd the model and to close it (to avoid holes in the model)
    Input: Image
    Output: File Name of the padded Image
    """
    # Size of the padding
    # Add 50 pixels around each dimension X,Y and 10 around Z
    padding_size = [50, 50, 10]

    padded_image = sitk.ConstantPad(input_image, padding_size, [0]*input_image.GetDimension())
    # Save the image
    filename = "image_padded.nii.gz"
    sitk.WriteImage(padded_image, filename)

    return filename

def Write(vtkdata, output_name)-> None:
        """
        Function to write VTK files
        Input : vtk data and path of the output
        """
        outfilename = output_name
        polydatawriter = vtk.vtkPolyDataWriter()
        polydatawriter.SetFileName(outfilename)
        polydatawriter.SetInputData(vtkdata)
        polydatawriter.Write()
