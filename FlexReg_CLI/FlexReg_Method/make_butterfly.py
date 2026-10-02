import numpy as np
import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from FlexReg_Method.orientation import orientation
from FlexReg_Method.util import vtkMeanTeeth, ToothNoExist
from FlexReg_Method.propagation import Dilation



# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger

logger = get_logger("FlexReg_make_butterfly")


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

    def __call__(self, t) :
        x , y = self.x0 + self.a * t , self.y0 + self.b * t

        return np.array([x ,y])
    

def Bezier_bled(point1,point2,point3,pas):
    range = np.arange(0,1,pas)
    matrix_t = np.array([np.square( 1 - range) , 2*(1 - range)*range, np.square(range)]).T
    matrix_point = np.array([[point1],[point2],[point3]]).squeeze()
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
        adjust_posterior_left,
        index,
        shift_lr=0.0,
        shift_ap=0.0
         ):
    
  

    surf_tmp = vtk.vtkPolyData()
    surf_tmp.DeepCopy(surf)


    radius = 0.7

    centroidf = vtkMeanTeeth([tooth_anterior_right,tooth_anterior_left,tooth_posterior_right,tooth_posterior_left],property='Universal_ID')

    try :
        surf_tmp = orientation(surf_tmp,[[-0.5,-0.5,0],[0,0,0],[0.5,-0.5,0]],
                                   ['3','5','12','14'])
        centroid = centroidf(surf_tmp)

    except ToothNoExist as error:
        logger.error(f' Error {error}')
        return
    
    ratio_anterior_left = (1 - ratio_anterior_left)/2
    ratio_anterior_right = (1 - ratio_anterior_right)/2
    ratio_posterior_left = (1 - ratio_posterior_left)/2
    ratio_posterior_right = (1 - ratio_posterior_right)/2

    V = torch.tensor(vtk_to_numpy(surf_tmp.GetPoints().GetData())).to(torch.float32)
    F = torch.tensor(vtk_to_numpy(surf_tmp.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(torch.int64)

    # Same vector on the four centroids : each landmark is an affine
    # combination of two of them with weights summing to 1, so the shift comes
    # back out of the interpolation and the patch is translated as a whole,
    # with its shape untouched.
    shift = np.array([shift_lr, shift_ap, 0], dtype=np.float32)

    c_ar = centroid[str(tooth_anterior_right)] + shift + np.array([0, adjust_anterior_right, 0], dtype=np.float32)
    c_al = centroid[str(tooth_anterior_left)] + shift + np.array([0, adjust_anterior_left, 0], dtype=np.float32)
    c_pr = centroid[str(tooth_posterior_right)] + shift + np.array([0, - adjust_posterior_right, 0], dtype=np.float32)
    c_pl = centroid[str(tooth_posterior_left)] + shift + np.array([0, -adjust_posterior_left, 0], dtype=np.float32)

    landmark_anterior_right = (1 - ratio_anterior_right) * c_ar + ratio_anterior_right * c_al
    landmark_posterior_right = (1 - ratio_posterior_right) * c_pr + ratio_posterior_right * c_pl
    
    landmark_anterior_left = (1 - ratio_anterior_left) * c_al + ratio_anterior_left * c_ar
    landmark_posterior_left = (1 - ratio_posterior_left) * c_pl + ratio_posterior_left * c_pr
    
    landmark_middle_posterior = (landmark_posterior_left + landmark_posterior_right) / 2
    middle = (landmark_anterior_left + landmark_anterior_right + landmark_posterior_left + landmark_posterior_right) / 4

    #rectangle limit
    t = np.arange(0,1,0.01)
    haut_seg = Segment2D(landmark_anterior_left,landmark_anterior_right)
    haut_seg = torch.tensor(haut_seg(t)).t().to(torch.float32)
    
    dis = torch.cdist(haut_seg,V[:,:2])
    arg_haut_seg = torch.unique(torch.argwhere(dis < radius).squeeze()[:,1])


    bas_seg = Segment2D(landmark_posterior_left,landmark_posterior_right)
    bas_seg = torch.tensor(bas_seg(t)).t().to(torch.float32)
    dis = torch.cdist(bas_seg,V[:,:2])
    arg_bas_seg = torch.unique(torch.argwhere(dis < radius).squeeze()[:,1])

    def compute_bezier_patch(start, middle, end, V, radius):
        bezier = Bezier_bled(start[:2], middle[:2], end[:2], 0.01)
        v_dir = (end[:2] - start[:2])
        v_norm = np.linalg.norm(v_dir)
        v_unit = v_dir / (v_norm + 1e-6)
        
        v_bezier = bezier - start[:2]
        proj = np.dot(v_bezier, v_unit)
        bezier_proj = np.outer(proj, v_unit) + start[:2]
        sym = 2 * bezier_proj - bezier
        
        dist = torch.cdist(torch.tensor(sym, dtype=torch.float32), V[:,:2])
        return torch.argwhere(dist < radius)[:, 1]

    arg_bezier_right = compute_bezier_patch(landmark_posterior_right, landmark_middle_posterior, landmark_anterior_right, V, radius)
    arg_bezier_left = compute_bezier_patch(landmark_posterior_left, landmark_middle_posterior, landmark_anterior_left, V, radius)

    v_label = torch.zeros((V.shape[0]))
    v_label[arg_haut_seg] = 1
    v_label[arg_bas_seg] = 1
    v_label[arg_bezier_right] = 1
    v_label[arg_bezier_left] = 1

    dist = torch.cdist(torch.tensor(middle[:2]).unsqueeze(0),V[:,:2]).squeeze()
    middle_arg = torch.argmin(dist)
    v_label = Dilation(middle_arg,F,v_label,surf_tmp)



    v_labels_prediction = numpy_to_vtk(v_label.cpu().numpy())
    v_labels_prediction.SetName(f'Butterfly{index}')



    surf.GetPointData().AddArray(v_labels_prediction)