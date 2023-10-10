import numpy as np
from Method.util import vtkMeanTeeth
import vtk
from Method.transformation import RotationMatrix, TransformSurf

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



def orientation(source,target,landmarks):


    left =landmarks[0]
    middle1 = landmarks[1]
    middle2 = landmarks[2]
    right = landmarks[3]

         
    meanTeeth = vtkMeanTeeth([int(left),int(middle1),int(middle2),int(right)],property='Universal_ID')
    mean_source = meanTeeth(source)

    left_source, middle1_source, middle2_source , right_source = mean_source[left], mean_source[middle1], mean_source[middle2],mean_source[right]
    left_target, middle_target , right_target = np.array(target[0]), np.array(target[1]), np.array(target[2])

    middle_source = (middle1_source + middle2_source) / 2


    normal_source, direction_source = make_vector([right_source,left_source],middle_source)
    normal_target , direction_target = make_vector([right_target,left_target],middle_target)



    dt = np.dot(normal_source,normal_target)
    if dt > 1.0 :
        dt = 1.0

    angle_normal = np.arccos(dt)


    normal_normal = cross(normal_source,normal_target)



    matrix_normal = RotationMatrix(normal_normal,angle_normal)
    
    
    
    direction_source = np.matmul(matrix_normal,direction_source.T).T
    direction_source = direction_source / np.linalg.norm(direction_source)

    
    direction_normal = cross(direction_source,direction_target)

    dt = np.dot(direction_source,direction_target)
    if dt > 1.0:
        dt = 1.0

    angle_direction = np.arccos(dt)
    matrix_direction = RotationMatrix(direction_normal ,angle_direction)



    matrix = np.matmul(matrix_direction, matrix_normal)

    left_source = np.matmul(matrix,left_source)   
    middle_source = np.matmul(matrix,middle_source)
    right_source = np.matmul(matrix,right_source)

    mean_source = np.mean(np.array([left_source,middle_source,right_source]),axis=0)
    mean_target = np.mean(np.array([left_target, middle_target, right_target]),axis=0)

    mean = (mean_target- mean_source)



    

    matrix = np.concatenate((matrix,np.array([mean]).T),axis=1)
    matrix = np.concatenate((matrix,np.array([[0,0,0,1]])),axis=0)


    # matrix = np.matmul(matrix,matrix_translation)



    



    output = vtk.vtkPolyData()
    output.DeepCopy(source)

    
    output = TransformSurf(output,matrix)


    return output