import shutil
import SimpleITK as sitk
import numpy as np
import argparse
import os
import glob
import sys
import csv

Spacing = []

def resample_fn(img, args):
    output_size = args['size'] 
    fit_spacing = args['fit_spacing']
    iso_spacing = args['iso_spacing']
    pixel_dimension = args['pixel_dimension']
    center = args['center']


    # if(pixel_dimension == 1):
    #     zeroPixel = 0
    # else:
    #     zeroPixel = np.zeros(pixel_dimension)

    if args['linear']:
        InterpolatorType = sitk.sitkLinear
    else:
        InterpolatorType = sitk.sitkNearestNeighbor

    

    spacing = img.GetSpacing()  
    size = img.GetSize()

    output_origin = img.GetOrigin()
    output_size = [si if o_si == -1 else o_si for si, o_si in zip(size, output_size)]
    # print(output_size)

    if(fit_spacing):
        output_spacing = [sp*si/o_si for sp, si, o_si in zip(spacing, size, output_size)]
    else:
        output_spacing = spacing

    if(iso_spacing):
        output_spacing_filtered = [sp for si, sp in zip(args['size'], output_spacing) if si != -1]
        # print(output_spacing_filtered)
        max_spacing = np.max(output_spacing_filtered)
        output_spacing = [sp if si == -1 else max_spacing for si, sp in zip(args['size'], output_spacing)]
        # print(output_spacing)

    if(args['spacing'] is not None):
        if isinstance(args['spacing'],float):
            output_spacing = [args['spacing'],args['spacing'],args['spacing']]
        else:
            output_spacing = args['spacing']

    if(args['origin'] is not None):
        output_origin = args['origin']

    if(center):
        output_physical_size = np.array(output_size)*np.array(output_spacing)
        input_physical_size = np.array(size)*np.array(spacing)
        output_origin = np.array(output_origin) - (output_physical_size - input_physical_size)/2.0

    if(args['direction']):
        output_direction = np.identity(3).flatten()
    else:
        output_direction = img.GetDirection()

    # print("Input size:", size)
    # print("Input spacing:", spacing)
    # print("Output size:", output_size)
    # print("Output spacing:", output_spacing)
    Spacing.append(output_spacing)
    # print("Output origin:", output_origin)

    resampleImageFilter = sitk.ResampleImageFilter()
    resampleImageFilter.SetInterpolator(InterpolatorType)   
    resampleImageFilter.SetOutputSpacing(output_spacing)
    resampleImageFilter.SetSize(output_size)
    resampleImageFilter.SetOutputDirection(output_direction)
    resampleImageFilter.SetOutputOrigin(output_origin)
    # resampleImageFilter.SetDefaultPixelValue(zeroPixel)
    

    return resampleImageFilter.Execute(img)


def Resample(img_filename, args):

    output_size = args['size'] 
    fit_spacing = args['fit_spacing']
    iso_spacing = args['iso_spacing']
    img_dimension = args['image_dimension']
    pixel_dimension = args['pixel_dimension']
    center = args['center']

    # print("Reading:", img_filename) 
    img = sitk.ReadImage(img_filename)

    return resample_fn(img, args)



def main(args):

    filenames = []
    if(args['img']):
        fobj = {}
        fobj["img"] = args['img']
        fobj["out"] = args['out']
        filenames.append(fobj)
    elif(args['dir']):
        out_dir = args['out']
        normpath = os.path.normpath("/".join([args['dir'], '**', '*']))
        FROM,WHERE = [],[]
        for img in glob.iglob(normpath, recursive=True):
            if os.path.isfile(img) and True in [ext in img for ext in [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".DCM", ".jpg", ".png", 'gipl.gz']]:
                fobj = {}
                fobj["img"] = img
                fobj["out"] = os.path.normpath(out_dir + "/" + '_'.join(img.replace(args['dir'], '').split('_')).split('.')[0]+'.nii.gz')
                if args['out_ext'] is not None:
                    out_ext = args['out_ext']
                    if out_ext[0] != ".":
                        out_ext = "." + out_ext
                    fobj["out"] = os.path.splitext(fobj["out"])[0] + out_ext
                if not os.path.exists(os.path.dirname(fobj["out"])):
                    os.makedirs(os.path.dirname(fobj["out"]))
                if not os.path.exists(fobj["out"]) or args['ow']:
                    filenames.append(fobj)
            if os.path.isfile(img) and True in [ext in img for ext in ["json"]]:
                FROM.append(img)
                WHERE.append(os.path.normpath(out_dir + "/" + img.replace(args['dir'], '')))
    else:
        raise "Set img or dir to resample!"

    if(args['rgb']):
        if(args['pixel_dimension'] == 3):
            print("Using: RGB type pixel with unsigned char")
        elif(args['pixel_dimension'] == 4):
            print("Using: RGBA type pixel with unsigned char")
        else:
            print("WARNING: Pixel size not supported!")

    if args['ref'] is not None:
        print(args['ref'])
        ref = sitk.ReadImage(args['ref'])
        args['size'] = ref.GetSize()
        args['spacing'] = ref.GetSpacing()
        args['origin'] = ref.GetOrigin()
    
    for fobj in filenames:

        if not os.path.exists(fobj["out"]):

            try:
                if "ref" in fobj and fobj["ref"] is not None:
                    ref = sitk.ReadImage(fobj["ref"])
                    args['size'] = ref.GetSize()
                    args['spacing'] = ref.GetSpacing()
                    args['origin'] = ref.GetOrigin()

                if args['size'] is not None:
                    img = Resample(fobj["img"], args)
                else:
                    img = sitk.ReadImage(fobj["img"])
                # print(len(Spacing))

                if args['spacing'] is not None:
                    #print("Writing:", fobj["out"])
                    writer = sitk.ImageFileWriter()
                    writer.SetFileName(fobj["out"])
                    writer.UseCompressionOn()
                    writer.Execute(img)

                # print("Size of file: {:.2f}kB".format(os.path.getsize(fobj["out"], ) / 1024))
                
            except Exception as e:
                print(e, file=sys.stderr)

def PreASOResample(data_dir,out_dir,spacing):
    
    # parser = argparse.ArgumentParser(description='Resample an image', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # in_group = parser.add_mutually_exclusive_group(required=False)

    # in_group.add_argument('--img', type=str, help='image to resample',default=None)
    # in_group.add_argument('--dir', type=str, help='Directory with image to resample',default=data_dir)
    # in_group.add_argument('--csv', type=str, help='CSV file with column img with paths to images to resample')

    # csv_group = parser.add_argument_group('CSV extra parameters')
    # csv_group.add_argument('--csv_column', type=str, default='image', help='CSV column name (Only used if flag csv is used)')
    # csv_group.add_argument('--csv_root_path', type=str, default='', help='Replaces a root path directory to empty, this is use to recreate a directory structure in the output directory, otherwise, the output name will be the name in the csv (only if csv flag is used)')

    # transform_group = parser.add_argument_group('Transform parameters')
    # transform_group.add_argument('--ref', type=str, help='Reference image. Use an image as reference for the resampling', default=None)
    # transform_group.add_argument('--size', nargs="+", type=int, help='Output size, -1 to leave unchanged',default=[128,128,128])
    # transform_group.add_argument('--spacing', nargs="+", type=float, default=scaling, help='Use a pre defined spacing')
    # transform_group.add_argument('--origin', nargs="+", type=float, default=None, help='Use a pre defined origin')
    # transform_group.add_argument('--linear', type=bool, help='Use linear interpolation.', default=False)
    # transform_group.add_argument('--center', type=bool, help='Center the image in the space', default=True)
    # transform_group.add_argument('--fit_spacing', type=bool, help='Fit spacing to output', default=True)
    # transform_group.add_argument('--iso_spacing', type=bool, help='Same spacing for resampled output', default=True)
    # transform_group.add_argument('--direction', type=bool, help='Set the direction of the output to the identity matrix', default=True)

    # img_group = parser.add_argument_group('Image parameters')
    # img_group.add_argument('--image_dimension', type=int, help='Image dimension', default=2)
    # img_group.add_argument('--pixel_dimension', type=int, help='Pixel dimension', default=1)
    # img_group.add_argument('--rgb', type=bool, help='Use RGB type pixel', default=False)

    # out_group = parser.add_argument_group('Ouput parameters')
    # out_group.add_argument('--ow', type=int, help='Overwrite', default=1)
    # out_group.add_argument('--out', type=str, help='Output image/directory', default=out_dir)
    # out_group.add_argument('--out_ext', type=str, help='Output extension type', default=None)
    # out_group.add_argument('--landmarks', type=bool, help="Copy landmark files to output directory", default=False)

    # args = parser.parse_args()

    args = {'img':None,
            'dir':data_dir,
            'ref':None,
            'size':[128,128,128],
            'spacing':spacing,
            'origin':None,
            'linear':False,
            'center':True,
            'fit_spacing':True,
            'iso_spacing':True,
            'direction':True,
            'image_dimension':2,
            'pixel_dimension':1,
            'rgb':False,
            'ow':1,
            'out':out_dir,
            'out_ext':None,
            }

    main(args)