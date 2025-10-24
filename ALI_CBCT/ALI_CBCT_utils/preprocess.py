import os
import numpy as np
import itk
import SimpleITK as sitk
import dicom2nifti

from ALI_CBCT_utils.io import search

def CorrectHisto(filepath,outpath,min_porcent=0.01,max_porcent = 0.95,i_min=-1500, i_max=4000):

    print("Correcting scan contrast :", filepath)
    input_img = sitk.ReadImage(filepath)
    input_img = sitk.Cast(input_img, sitk.sitkFloat32)
    img = sitk.GetArrayFromImage(input_img)


    img_min = np.min(img)
    img_max = np.max(img)
    img_range = img_max - img_min
    # print(img_min,img_max,img_range)

    definition = 1000
    histo = np.histogram(img,definition)
    cum = np.cumsum(histo[0])
    cum = cum - np.min(cum)
    cum = cum / np.max(cum)

    res_high = list(map(lambda i: i> max_porcent, cum)).index(True)
    res_max = (res_high * img_range)/definition + img_min

    res_low = list(map(lambda i: i> min_porcent, cum)).index(True)
    res_min = (res_low * img_range)/definition + img_min

    res_min = max(res_min,i_min)
    res_max = min(res_max,i_max)


    # print(res_min,res_min)

    img = np.where(img > res_max, res_max,img)
    img = np.where(img < res_min, res_min,img)

    output = sitk.GetImageFromArray(img)
    output.SetSpacing(input_img.GetSpacing())
    output.SetDirection(input_img.GetDirection())
    output.SetOrigin(input_img.GetOrigin())
    output = sitk.Cast(output, sitk.sitkInt16)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(output)
    return output

def ResampleImage(input,size,spacing,origin,direction,interpolator,VectorImageType):
    ResampleType = itk.ResampleImageFilter[VectorImageType, VectorImageType]

    resampleImageFilter = ResampleType.New()
    resampleImageFilter.SetOutputSpacing(spacing.tolist())
    resampleImageFilter.SetOutputOrigin(origin)
    resampleImageFilter.SetOutputDirection(direction)
    resampleImageFilter.SetInterpolator(interpolator)
    resampleImageFilter.SetSize(size)
    resampleImageFilter.SetInput(input)
    resampleImageFilter.Update()

    resampled_img = resampleImageFilter.GetOutput()
    return resampled_img

def SetSpacing(filepath,output_spacing=[0.5, 0.5, 0.5],outpath=-1):
    """
    Set the spacing of the image at the wanted scale

    Parameters
    ----------
    filePath
     path of the image file
    output_spacing
     whanted spacing of the new image file (default : [0.5, 0.5, 0.5])
    outpath
     path to save the new image
    """

    print("Resample :", filepath, ", with spacing :", output_spacing)
    img = itk.imread(filepath)
    spacing = np.array(img.GetSpacing())
    output_spacing = np.array(output_spacing)

    if not np.array_equal(spacing,output_spacing):

        size = itk.size(img)
        scale = spacing/output_spacing

        output_size = (np.array(size)*scale).astype(int).tolist()
        output_origin = img.GetOrigin()

        #Find new origin
        output_physical_size = np.array(output_size)*np.array(output_spacing)
        input_physical_size = np.array(size)*spacing
        output_origin = np.array(output_origin) - (output_physical_size - input_physical_size)/2.0

        img_info = itk.template(img)[1]
        pixel_type = img_info[0]
        pixel_dimension = img_info[1]

        VectorImageType = itk.Image[pixel_type, pixel_dimension]

        if True in [seg in os.path.basename(filepath) for seg in ["seg","Seg"]]:
            InterpolatorType = itk.NearestNeighborInterpolateImageFunction[VectorImageType, itk.D]
            # print("Rescale Seg with spacing :", output_spacing)
        else:
            InterpolatorType = itk.LinearInterpolateImageFunction[VectorImageType, itk.D]
            # print("Rescale Scan with spacing :", output_spacing)

        interpolator = InterpolatorType.New()
        resampled_img = ResampleImage(img,output_size,output_spacing,output_origin,img.GetDirection(),interpolator,VectorImageType)

        if outpath != -1:
            itk.imwrite(resampled_img, outpath)
        return resampled_img

    else:
        # print("Already at the wanted spacing")
        if outpath != -1:
            itk.imwrite(img, outpath)
        return img

def convertdicom2nifti(input_folder,output_folder=None):
    patients_folders = [folder for folder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder,folder)) and folder != 'NIFTI']

    if output_folder is None:
        output_folder = os.path.join(input_folder,'NIFTI')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for patient in patients_folders:
        if not os.path.exists(os.path.join(output_folder,patient+".nii.gz")):
            print("Converting patient: {}...".format(patient))
            current_directory = os.path.join(input_folder,patient)
            try:
                reader = sitk.ImageSeriesReader()
                sitk.ProcessObject_SetGlobalWarningDisplay(False)
                dicom_names = reader.GetGDCMSeriesFileNames(current_directory)
                reader.SetFileNames(dicom_names)
                image = reader.Execute()
                sitk.ProcessObject_SetGlobalWarningDisplay(True)
                sitk.WriteImage(image, os.path.join(output_folder,os.path.basename(current_directory)+'.nii.gz'))
            except RuntimeError:
                dicom2nifti.convert_directory(current_directory,output_folder)
                nifti_file = search(output_folder,'nii.gz')['nii.gz'][0]
                os.rename(nifti_file,os.path.join(output_folder,patient+".nii.gz"))