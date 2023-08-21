import numpy as np
import os
import SimpleITK as sitk


def ReadMatrix(path):
    fname, extension = os.path.splitext(os.path.basename(path))
    extension = extension.lower()

    if extension==".npy" :
        matrix = np.load(path)
    else :
        transform = sitk.ReadTransform(path)
        transform = transform.GetInverse()
        rotation = transform.GetMatrix()
        translation = transform.GetTranslation()

        matrix = np.array(rotation).reshape(3,3)
        matrix = np.concatenate((matrix,np.array([translation]).T),axis=1)
        matrix = np.concatenate((matrix,np.array([[0,0,0,1]])),axis=0)

    return matrix
                