import numpy as np
import vtk
from .util import vtkMeanTeeth
from .transform import RotationMatrix, TransformSurf

import sys
import logging

# ===== Logging Configuration =====
logger = logging.getLogger("FlexReg_orientation")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

cross = lambda a,b: np.cross(a,b)

def make_vector(points2,point1):
    perpen = points2[1]-points2[0]
    perpen= perpen/np.linalg.norm(perpen)

    vector1 = points2[0] - point1
    vector1 = vector1/np.linalg.norm(vector1)

    vector2 = points2[1] - point1
    vector2 = vector2/np.linalg.norm(vector2)

    normal = cross(vector1,vector2)
    normal = normal/np.linalg.norm(normal)

    direction = cross(normal,perpen)
    direction = direction/np.linalg.norm(direction)
    return normal ,direction



def rotation_between(source_vector,target_vector):
    '''
    Rotation bringing source_vector onto target_vector.

    The dot product is clamped on both ends -- rounding can push it just past
    -1, where arccos returns NaN -- and the two degenerate cases are handled
    explicitly : when the vectors are already aligned or exactly opposed their
    cross product vanishes, and normalising that null axis would turn the whole
    matrix into NaN without raising anything.
    '''
    dt = float(np.clip(np.dot(source_vector,target_vector),-1.0,1.0))
    axis = cross(source_vector,target_vector)

    if np.linalg.norm(axis) < 1e-8:
        if dt > 0:
            return np.identity(3)
        # opposed : half a turn around any axis perpendicular to the source
        fallback = np.array([0.0,1.0,0.0]) if abs(source_vector[0]) > 0.9 else np.array([1.0,0.0,0.0])
        axis = cross(source_vector,fallback)

    return RotationMatrix(axis,np.arccos(dt))


def orientation_matrix(source, target, landmarks, label_array='Universal_ID'):
    '''
    Compute the 4x4 rigid matrix bringing source into the canonical frame
    described by target. Same computation as orientation_f, which now builds
    on it -- the matrix itself is needed to place the patch preview back in
    the scene without transforming the whole surface.
    '''

    left =landmarks[0]
    middle1 = landmarks[1]
    middle2 = landmarks[2]
    right = landmarks[3]

         
    # The label array is passed in rather than fixed: a mesh straight out of
    # Crown_Seg carries `PredictedID`, and hardcoding one name refused it.
    meanTeeth = vtkMeanTeeth([int(left), int(middle1), int(middle2), int(right)],
                             property=label_array)
    mean_source = meanTeeth(source)

    left_source, middle1_source, middle2_source , right_source = mean_source[left], mean_source[middle1], mean_source[middle2],mean_source[right]
    left_target, middle_target , right_target = np.array(target[0]), np.array(target[1]), np.array(target[2])

    middle_source = (middle1_source + middle2_source) / 2


    normal_source, direction_source = make_vector([right_source,left_source],middle_source)
    normal_target , direction_target = make_vector([right_target,left_target],middle_target)



    matrix_normal = rotation_between(normal_source,normal_target)



    direction_source = np.matmul(matrix_normal,direction_source.T).T
    direction_source = direction_source / np.linalg.norm(direction_source)


    matrix_direction = rotation_between(direction_source,direction_target)



    matrix = np.matmul(matrix_direction, matrix_normal)

    left_source = np.matmul(matrix,left_source)   
    middle_source = np.matmul(matrix,middle_source)
    right_source = np.matmul(matrix,right_source)

    mean_source = np.mean(np.array([left_source,middle_source,right_source]),axis=0)
    mean_target = np.mean(np.array([left_target, middle_target, right_target]),axis=0)

    mean = (mean_target- mean_source)



    

    matrix = np.concatenate((matrix,np.array([mean]).T),axis=1)
    matrix = np.concatenate((matrix,np.array([[0,0,0,1]])),axis=0)

    return matrix


def orientation_f(source,target,landmarks):

    matrix = orientation_matrix(source,target,landmarks)

    output = vtk.vtkPolyData()
    output.DeepCopy(source)


    output = TransformSurf(output,matrix)


    return output