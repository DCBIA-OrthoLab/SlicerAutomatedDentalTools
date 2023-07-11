from typing import Any
import torch
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from AREG_IOS_utils.net import MonaiUNetHRes
from AREG_IOS_utils.post_process import RemoveIslands, DilateLabel, ErodeLabel




class PredPatch:
    """
    PredPatch class is to predict/draw patch on the palate

    """
    def __init__(self,path_model) -> None:
        self.model = MonaiUNetHRes()
        self.model.load_state_dict(torch.load(path_model)['state_dict'])

        self.device = torch.device('cuda')
        self.model.to(self.device)
        self.model.eval()
        self.softmax= torch.nn.Softmax(dim=2)

    def __call__(self, batch,surf) -> Any:
        with torch.no_grad():

            out_channels = 2

            V, F, CN = batch

            V = V.cuda(non_blocking=True)
            F = F.cuda(non_blocking=True)
            CN = CN.cuda(non_blocking=True).to(torch.float32)
            CN = CN.unsqueeze(0)
            F = F.unsqueeze(0)
            V = V.unsqueeze(0)

            x, X, PF = self.model((V, F, CN))
            x = self.softmax(x*(PF>=0))  

            P_faces = torch.zeros(out_channels, F.shape[1]).to(self.device)
            V_labels_prediction = torch.zeros(V.shape[1]).to(self.device).to(torch.int64)

            PF = PF.squeeze()
            x = x.squeeze(0)


            for pf, pred in zip(PF, x):
                P_faces[:, pf] += pred


            P_faces = torch.argmax(P_faces, dim=0)

            faces_pid0 = F[0,:,0]
            V_labels_prediction[faces_pid0] = P_faces
            
            V_labels_prediction = torch.where(V_labels_prediction >= 1, 1, 0)

            V_labels_prediction = numpy_to_vtk(V_labels_prediction.cpu().numpy())
            V_labels_prediction.SetName('Butterfly')
            surf.GetPointData().AddArray(V_labels_prediction)


            #Post Process
            #fill the holes in patch
            RemoveIslands(surf,V_labels_prediction,33,500, ignore_neg1=True)
            for label in range(2):
                RemoveIslands(surf,V_labels_prediction, label, 200, ignore_neg1=True)




            for label in range(1,2):
                DilateLabel(surf,V_labels_prediction, label, iterations=2, dilateOverTarget=False, target = None)
                ErodeLabel(surf,V_labels_prediction, label, iterations=2, target=None) 

        return surf