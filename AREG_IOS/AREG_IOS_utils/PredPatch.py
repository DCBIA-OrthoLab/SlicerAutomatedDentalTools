from typing import Any
import torch
from vtk.util.numpy_support import numpy_to_vtk
from AREG_IOS_utils.net import MonaiUNetHRes
from AREG_IOS_utils.post_process import RemoveIslands, DilateLabel, ErodeLabel
# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger

logger = get_logger("AREG_IOS_PredPatch")


class PredPatch:
    """
    PredPatch class is to predict/draw patch on the palate

    """

    def __init__(self, path_model) -> None:
        self.model = MonaiUNetHRes()
        # `weights_only=False` because this is a Lightning checkpoint, not a
        # bare state dict: it carries the training metadata Lightning pickles
        # beside the weights. torch 2.6 flipped this default to True, and the
        # load has refused ever since with "Weights only load failed" -- the
        # file did not change, the default did. The path is the model this
        # project ships, named by the deployment, not something a caller sends.
        self.model.load_state_dict(
            torch.load(path_model, weights_only=False)["state_dict"])

        self.device = torch.device("cuda")
        self.model.to(self.device)
        self.model.eval()
        self.softmax = torch.nn.Softmax(dim=2)

    def __call__(self, batch, surf) -> Any:
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
            x = self.softmax(x * (PF >= 0))

            p_faces = torch.zeros(out_channels, F.shape[1]).to(self.device)
            v_labels_prediction = (
                torch.zeros(V.shape[1]).to(self.device).to(torch.int64)
            )

            PF = PF.squeeze()
            x = x.squeeze(0)

            for pf, pred in zip(PF, x):
                p_faces[:, pf] += pred

            p_faces = torch.argmax(p_faces, dim=0)

            faces_pid0 = F[0, :, 0]
            v_labels_prediction[faces_pid0] = p_faces

            v_labels_prediction = torch.where(v_labels_prediction >= 1, 1, 0)

            v_labels_prediction = numpy_to_vtk(v_labels_prediction.cpu().numpy())
            v_labels_prediction.SetName("Butterfly")
            surf.GetPointData().AddArray(v_labels_prediction)

            # Post Process
            # fill the holes in patch
            RemoveIslands(surf, v_labels_prediction, 33, 500, ignore_neg1=True)
            for label in range(2):
                RemoveIslands(surf, v_labels_prediction, label, 200, ignore_neg1=True)

            for label in range(1, 2):
                DilateLabel(
                    surf,
                    v_labels_prediction,
                    label,
                    iterations=2,
                    dilateOverTarget=False,
                    target=None,
                )
                ErodeLabel(surf, v_labels_prediction, label, iterations=2, target=None)

        return surf
