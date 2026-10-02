import torch
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np

from FlexReg_Method.propagation import Dilation


# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger

logger = get_logger("FlexReg_draw")

def drawPatch(outline_points: list,polydata,mid,index:int):
    step = 0.2
    radius = 0.5
    radius = 1.1
    P0 = torch.tensor(np.array(outline_points)).unsqueeze(0).cuda()
    P1 = torch.tensor(np.array(outline_points[1:] + [outline_points[0]])).unsqueeze(0).cuda()


    T = torch.arange(0,1+step,step).unsqueeze(0).unsqueeze(0).permute(2,1,0).cuda()

    P = (1-T)*P0 + T*P1

    pshape= P.shape

    P = P.view(pshape[0]*pshape[1],3)

         

    V = torch.tensor(vtk_to_numpy(polydata.GetPoints().GetData())).to(torch.float32).cuda()
    F = torch.tensor(vtk_to_numpy(polydata.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(torch.int64).cuda()

    dist = torch.cdist(P,V)
    arg_outline = torch.argwhere(dist < radius)[:,1]
    v_label = torch.zeros((V.shape[0])).cuda()
    v_label[arg_outline] = 1

    mid = torch.tensor(mid).unsqueeze(0).cuda()
    dist_mid_vertex = torch.cdist(mid,V)
    arg_midpoint_min = torch.argmin(dist_mid_vertex)
    v_label = Dilation(arg_midpoint_min,F,v_label,polydata)

    v_labels_prediction = numpy_to_vtk(v_label.cpu().numpy())
    v_labels_prediction.SetName(f'Butterfly{index}')

    polydata.GetPointData().AddArray(v_labels_prediction)
    
