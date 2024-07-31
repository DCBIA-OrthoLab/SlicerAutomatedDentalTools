from typing import Any
import torch
from collections import deque
import vtk
import numpy as np

# def wrap_error(func):
#     nbloop =1
#     def wrapper(arg_point,F):
#         error = False
#         nbloop = 1
#         neighbours = torch.tensor([]).cuda()
#         while error :
#             arg_point_loop = 
#             try :
#                 neighbours_ = func(arg_point,F)
#             except RuntimeError :
#                 error = True

#             neighbours = torch.cat([neighbours,neighbours_])

#         return neighbours

#     return wrapper


# class WrappError:
#     def __init__(self,func) -> None:
#         self._func = func
#         self.nbloop = 1

#     def __call__(self,arg_point,F) :
#         error = False
#         neighbours = torch.tensor([]).cuda()
#         while error :
#             arg_point_loop = 
#             try :
#                 neighbours_ = func(arg_point,F)
#             except RuntimeError :
#                 error = True

#             neighbours = torch.cat([neighbours,neighbours_])

#         return neighbours
    

#     def limitTest()


def Difference(t1,t2):
    t1 = t1.unsqueeze(0).expand(len(t2),-1)
    t2 = t2.unsqueeze(1)
    d = torch.count_nonzero(t1 -t2,dim=-1)
    arg = torch.argwhere(d == t1.shape[1])
    dif = torch.unique(t2[arg])
    return dif


def Neighbours(arg_point,F):
    neighbours = torch.tensor([]).cuda()
    F2 = F.unsqueeze(0).expand(len(arg_point),-1,-1)
    arg_point = arg_point.unsqueeze(1).unsqueeze(2)
    arg = torch.argwhere((F2-arg_point) == 0)

    neighbours = torch.unique(F[arg[:,1],:])
    return neighbours



# def GetNeighbors(vtkdata, pid):

#     if isinstance(pid, torch.Tensor):
#         pid = pid.item()
#     elif isinstance(pid, np.ndarray):
#         pid = int(pid[0])

#     cells_id = vtk.vtkIdList()
#     vtkdata.GetPointCells(pid, cells_id)
#     neighbor_pids = []

#     for ci in range(cells_id.GetNumberOfIds()):
#         points_id_inner = vtk.vtkIdList()
#         vtkdata.GetCellPoints(cells_id.GetId(ci), points_id_inner)
#         for pi in range(points_id_inner.GetNumberOfIds()):
#             pid_inner = points_id_inner.GetId(pi)
#             if pid_inner != pid:
#                 neighbor_pids.append(pid_inner)

#     return np.unique(neighbor_pids).tolist()


def GetNeighbors(vtkdata, pids_tensor):
    all_neighbor_pids = []

    # Convertir le tensor en une liste d'entiers
    pids_list = pids_tensor.tolist()

    for pid in pids_list:
        cells_id = vtk.vtkIdList()
        vtkdata.GetPointCells(pid, cells_id)

        for ci in range(cells_id.GetNumberOfIds()):
            points_id_inner = vtk.vtkIdList()
            vtkdata.GetCellPoints(cells_id.GetId(ci), points_id_inner)
            for pi in range(points_id_inner.GetNumberOfIds()):
                pid_inner = points_id_inner.GetId(pi)
                if pid_inner != pid:
                    all_neighbor_pids.append(pid_inner)

    # Rendre unique tous les indices de voisins
    unique_neighbors = np.unique(all_neighbor_pids).tolist()
    return torch.tensor(unique_neighbors).cuda().to(torch.int64)




def Dilation(arg_point,F,texture,surf):
    arg_point = torch.tensor([arg_point]).cuda().to(torch.int64)
    F = F.cuda()
    texture = texture.cuda()
    neighbour = Neighbours(arg_point,F)
    arg_texture = torch.argwhere(texture == 1).squeeze()
    # dif = NoIntersection(arg_texture,neighbour)
    dif = neighbour.to(torch.int64)
    dif  = Difference(arg_texture,dif)

    dif_queue = [Neighbours(arg_point,F).to(torch.int64)]
    

    nmb_treatment = 1000

    while dif_queue :  # La boucle continue tant que l'une des files d'attente n'est pas vide
        new_neighbour_batch = []
        while dif_queue:
            current_dif = dif_queue.pop(0)
            if current_dif.numel() > nmb_treatment:
                dif_queue.append(current_dif[nmb_treatment:])
                current_dif = current_dif[:nmb_treatment]
            texture[current_dif] = 1
            new_neighbour_batch.append(GetNeighbors(surf,current_dif))
        
        arg_texture = torch.argwhere(texture == 1).squeeze()
        
        while new_neighbour_batch:
            current_neighbours = new_neighbour_batch.pop(0)
            if current_neighbours.numel() > nmb_treatment:
                new_neighbour_batch.append(current_neighbours[nmb_treatment:])
                current_neighbours = current_neighbours[:nmb_treatment]
            dif = Difference(arg_texture, current_neighbours.to(torch.int64))
            if dif.numel() > 0:
                dif_queue.append(dif)
    return texture