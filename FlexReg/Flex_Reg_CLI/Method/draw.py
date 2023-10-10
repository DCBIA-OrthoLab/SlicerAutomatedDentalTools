import torch
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np

from Method.propagation import Dilation

def drawPatch(outlinePoints: list,polydata,mid):
    step = 0.2
    radius = 0.5 # celui de Nathan
    radius = 1.1
    P0 = torch.tensor(np.array(outlinePoints)).unsqueeze(0).cuda()
    P1 = torch.tensor(np.array(outlinePoints[1:] + [outlinePoints[0]])).unsqueeze(0).cuda()


    T = torch.arange(0,1+step,step).unsqueeze(0).unsqueeze(0).permute(2,1,0).cuda()

    P = (1-T)*P0 + T*P1

    Pshape= P.shape

    P = P.view(Pshape[0]*Pshape[1],3)

    V = torch.tensor(vtk_to_numpy(polydata.GetPoints().GetData())).to(torch.float32).cuda()
    F = torch.tensor(vtk_to_numpy(polydata.GetPolys().GetData()).reshape(-1, 4)[:,1:]).to(torch.int64).cuda()

    dist = torch.cdist(P,V)
    arg_outline = torch.argwhere(dist < radius)[:,1]
    V_label = torch.zeros((V.shape[0])).cuda()
    V_label[arg_outline] = 1

    mid = torch.tensor(mid).unsqueeze(0).cuda()
    dist_mid_vertex = torch.cdist(mid,V)
    arg_midpoint_min = torch.argmin(dist_mid_vertex)
    V_label = Dilation(arg_midpoint_min,F,V_label,polydata)

    V_labels_prediction = numpy_to_vtk(V_label.cpu().numpy())
    V_labels_prediction.SetName('Butterfly')

    polydata.GetPointData().AddArray(V_labels_prediction)
    
