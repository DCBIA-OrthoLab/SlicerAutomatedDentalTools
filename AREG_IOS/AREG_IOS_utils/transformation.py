import numpy as np
import vtk
def RotationMatrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    Parameters
    ----------
    axis : np.array
        Axis of rotation
    theta : float
        Angle of rotation in radians
    
    Returns
    -------
    np.array
        Rotation matrix
    """

    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])




def TransformSurf(surf,matrix):
    assert isinstance(surf,vtk.vtkPolyData)
    surf_copy = vtk.vtkPolyData()
    surf_copy.DeepCopy(surf)
    surf = surf_copy

    transform = vtk.vtkTransform()
    transform.SetMatrix(np.reshape(matrix,16))
    surf = RotateTransform(surf,transform)

    return surf



def RotateTransform(surf, transform):

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(surf)
    transformFilter.Update()
    return transformFilter.GetOutput()




def TransformSurf(surf,matrix):
    assert isinstance(surf,vtk.vtkPolyData)
    surf_copy = vtk.vtkPolyData()
    surf_copy.DeepCopy(surf)
    surf = surf_copy

    transform = vtk.vtkTransform()
    transform.SetMatrix(np.reshape(matrix,16))
    surf = RotateTransform(surf,transform)

    return surf




def TransformList(input,matrix):
    type = np.array
    if isinstance(input,list):
        input = np.array(input)
        type = list

    a = np.ones((input.shape[0],1))

    input = np.hstack((input,a))
    matrix = matrix[:3,:]
    input = np.matmul(matrix ,input.T).T

    if isinstance(type,list):
        input = input.tolist()

    return input



def ApplyTransform(input,transform):
    if isinstance(input,vtk.vtkPolyData):
        input = TransformSurf(input,transform)

    if isinstance(input,dict):
        input = TransformDict(input,transform)

    if isinstance(input,(list,np.ndarray)):
        input = TransformList(input,transform)

    return input



def TransformDict(source,transform):
    '''
    Apply a transform matrix to a set of landmarks
    
    Parameters
    ----------
    source : dict
        Dictionary of landmarks
    transform : np.array
        Transform matrix
    
    Returns
    -------
    source : dict
        Dictionary of transformed landmarks
    '''

    sourcee = source.copy()
    for key in sourcee.keys():
        sourcee[key] = transform @ np.append(sourcee[key],1)
        sourcee[key] = sourcee[key][:3]
    return sourcee



def ScaleSurf(surf, mean_arr = None, scale_factor = None):

    surf_copy = vtk.vtkPolyData()
    surf_copy.DeepCopy(surf)
    surf = surf_copy


    shapedatapoints = surf.GetPoints()

    #calculate bounding box
    mean_v = [0.0] * 3
    bounds_max_v = [0.0] * 3

    bounds = shapedatapoints.GetBounds()

    mean_v[0] = (bounds[0] + bounds[1])/2.0
    mean_v[1] = (bounds[2] + bounds[3])/2.0
    mean_v[2] = (bounds[4] + bounds[5])/2.0
    bounds_max_v[0] = max(bounds[0], bounds[1])
    bounds_max_v[1] = max(bounds[2], bounds[3])
    bounds_max_v[2] = max(bounds[4], bounds[5])

    shape_points = []
    for i in range(shapedatapoints.GetNumberOfPoints()):
        p = shapedatapoints.GetPoint(i)
        shape_points.append(p)
    shape_points = np.array(shape_points)
    
    #centering points of the shape
    if mean_arr is None:
        mean_arr = np.array(mean_v)
    # print("Mean:", mean_arr)
    shape_points = shape_points - mean_arr

    #Computing scale factor if it is not provided
    if(scale_factor is None):
        bounds_max_arr = np.array(bounds_max_v)
        scale_factor = 1/np.linalg.norm(bounds_max_arr - mean_arr)

    #scale points of the shape by scale factor
    # print("Scale:", scale_factor)
    shape_points_scaled = np.multiply(shape_points, scale_factor)

    #assigning scaled points back to shape
    for i in range(shapedatapoints.GetNumberOfPoints()):
       shapedatapoints.SetPoint(i, shape_points_scaled[i])    

    surf.SetPoints(shapedatapoints)

    return surf