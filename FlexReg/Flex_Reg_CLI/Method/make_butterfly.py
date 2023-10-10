import numpy as np
import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from Method.orientation import orientation
from Method.util import vtkMeanTeeth, ToothNoExist
from Method.propagation import Dilation





class Segment2D :
    def __init__(self,point1,point2,name_point1 =None , name_point2 = None) -> None:
        self.point1 = np.array(point1)
        self.point2 = np.array(point2)
        self.a = point2[0] - point1[0]
        self.b = point2[1] - point1[1]

        self.x0 = point1[0]
        self.y0 = point1[1] 

        self.name_point1 = name_point1
        self.name_point2 = name_point2

        # print(f' a : {self.a}, b : {self.b}, x0 : {self.x0}, y0 : {self.y0}')

    def __call__(self, t) :
        x , y = self.x0 + self.a * t , self.y0 + self.b * t

        return np.array([x ,y])
    

def Bezier_bled(point1,point2,point3,pas):
    range = np.arange(0,1,pas)
    matrix_t = np.array([np.square( 1 - range) , 2*(1 - range)*range, np.square(range)]).T
    matrix_point = np.array([[point1],[point2],[point3]]).squeeze()
    # print(f'shape matrix_t {matrix_t.shape}, matrix point {matrix_point.shape}')
    return np.matmul(matrix_t,matrix_point)



def butterflyPatch(surf,
            tooth_anterior_right,
         tooth_anterior_left,
         tooth_posterior_right,
         tooth_posterior_left,
        ratio_anterior_right,
        ratio_anterior_left,
        ratio_posterior_left,
        ratio_posterior_right,
        adjust_anterior_right,
        adjust_anterior_left,
        adjust_posterior_right,
        adjust_posterior_left
         ):
    
  

    surf_tmp = vtk.vtkPolyData()
    surf_tmp.DeepCopy(surf)


    radius = 0.7

    # centroidf = vtkMeanTeeth([6,11,3,14],property='Universal_ID')
    centroidf = vtkMeanTeeth([tooth_anterior_right,tooth_anterior_left,tooth_posterior_right,tooth_posterior_left],property='Universal_ID')

    # centroidf = vtkMeanTeeth([2,5,12,15,6,11,3,14],property='Universal_ID')
    try :
        surf_tmp = orientation(surf_tmp,[[-0.5,-0.5,0],[0,0,0],[0.5,-0.5,0]],
                                   ['3','5','12','14'])
        centroid = centroidf(surf_tmp)

    except ToothNoExist as error:
        print(f' Error {error}')
        # quit()
        return
    V = torch.tensor(vtk_to_numpy(surf_tmp.GetPoints().GetData())).to(torch.float32)
    F = torch.tensor(vtk_to_numpy(surf_tmp.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(torch.int64)

    centroid_anterior_right = centroid[str(tooth_anterior_right)] + np.array([0,adjust_anterior_right,0],dtype=np.float32)
    centroid_anterior_left = centroid[str(tooth_anterior_left)] + np.array([0,adjust_anterior_left,0],dtype=np.float32)


    centroid_posterior_rigth = centroid[str(tooth_posterior_right)] + np.array([0,adjust_posterior_right,0],dtype=np.float32)
    centroid_posterior_left = centroid[str(tooth_posterior_left)]+ np.array([0,adjust_posterior_left ,0],dtype=np.float32)


    landmark_anterior_left = (1-ratio_anterior_left) * centroid_anterior_right + ratio_anterior_left * centroid_anterior_left
    landmark_anterior_right = (1-ratio_anterior_right) * centroid_anterior_left + ratio_anterior_right * centroid_anterior_right

    landmark_posterior_left = (1-ratio_posterior_left) * centroid_posterior_rigth + ratio_posterior_left * centroid_posterior_left
    landmark_posterior_right = (1- ratio_posterior_right) * centroid_posterior_left + ratio_posterior_right * centroid_posterior_rigth
    landmark_middle_posterior = (landmark_posterior_left + landmark_posterior_right) / 2



    middle = (landmark_posterior_left + landmark_anterior_right) / 2



    #rectangle limit
    t = np.arange(0,1,0.01)
    haut_seg = Segment2D(landmark_anterior_left,landmark_anterior_right)
    haut_seg = torch.tensor(haut_seg(t)).t().to(torch.float32)
    # print(haut_seg)
    dis = torch.cdist(haut_seg,V[:,:2])
    arg_haut_seg = torch.unique(torch.argwhere(dis < radius).squeeze()[:,1])


    bas_seg = Segment2D(landmark_posterior_left,landmark_posterior_right)
    bas_seg = torch.tensor(bas_seg(t)).t().to(torch.float32)
    dis = torch.cdist(bas_seg,V[:,:2])
    arg_bas_seg = torch.unique(torch.argwhere(dis < radius).squeeze()[:,1])



    


    #bezier droite
    bezier = Bezier_bled(landmark_posterior_right[:2],landmark_middle_posterior[:2],landmark_anterior_right[:2],0.01)
    v_bezier = bezier - np.expand_dims(landmark_posterior_right[:2],axis=0)
    v_norm_bezier = np.expand_dims(np.linalg.norm(v_bezier, axis=1),axis=0).T
    # print(f"v_norm_bezier : {v_norm_bezier}")
    for i in range(len(v_norm_bezier)):
        if v_norm_bezier[i] == 0:
            v_norm_bezier[i] += 0.01

    v_bezier = v_bezier / v_norm_bezier

    v = np.expand_dims(landmark_anterior_right[:2] - landmark_posterior_right[:2], axis=0).T
    v_norm = np.linalg.norm(v)
    v = v / v_norm
    # print(f'v {v}')
    P = np.matmul(v , v.T)

    bezier_proj = ( P @ v_bezier.T).T *v_norm_bezier + landmark_posterior_right[:2]
    sym = 2*bezier_proj - bezier

    bezier = torch.tensor(sym,dtype=torch.float32)
    dist = torch.cdist(bezier,V[:,:2])
    arg_bezier = torch.argwhere(dist < radius)[:,1]




    #bezier gauche
    bezier2 = Bezier_bled(landmark_posterior_left[:2],landmark_middle_posterior[:2],landmark_anterior_left[:2],0.01)
    v_bezier = bezier2 - np.expand_dims(landmark_posterior_left[:2],axis=0)
    v_norm_bezier = np.expand_dims(np.linalg.norm(v_bezier, axis=1),axis=0).T
    for i in range(len(v_norm_bezier)):
        if v_norm_bezier[i] == 0:
            v_norm_bezier[i] += 0.01
    v_bezier = v_bezier / v_norm_bezier

    v = np.expand_dims(landmark_anterior_left[:2] - landmark_posterior_left[:2], axis=0).T
    v_norm = np.linalg.norm(v)
    v = v / v_norm
    # print(f'v {v}')
    P = np.matmul(v , v.T)

    bezier_proj = ( P @ v_bezier.T).T *v_norm_bezier + landmark_posterior_left[:2]
    sym = 2*bezier_proj - bezier2

    bezier2 = torch.tensor(sym,dtype=torch.float32)
    dist = torch.cdist(bezier2,V[:,:2])
    print(f'bezier2.shape {bezier2.shape}, V[:,;2].shape {V[:,:2].shape}')
    arg_bezier2 = torch.argwhere(dist < radius)[:,1]





    V_label = torch.zeros((V.shape[0]))
    V_label[arg_haut_seg] = 1
    V_label[arg_bas_seg] = 1
    V_label[arg_bezier] = 1
    V_label[arg_bezier2] = 1



    dist = torch.cdist(torch.tensor(middle[:2]).unsqueeze(0),V[:,:2]).squeeze()
    middle_arg = torch.argmin(dist)
    V_label = Dilation(middle_arg,F,V_label,surf_tmp)



    V_labels_prediction = numpy_to_vtk(V_label.cpu().numpy())
    V_labels_prediction.SetName('Butterfly')



    surf.GetPointData().AddArray(V_labels_prediction)




    # return surf




