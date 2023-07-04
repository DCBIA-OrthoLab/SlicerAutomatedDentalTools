import vtk
import numpy as np
import argparse
import sys
import os
import math







def ConnectedRegion(vtkdata, pid, labels, label, pid_visited):

	neighbor_pids = GetNeighborIds(vtkdata, pid, labels, label, pid_visited)
	all_connected_pids = [pid]
	all_connected_pids.extend(neighbor_pids)

	while len(neighbor_pids):
		npid = neighbor_pids.pop()
		next_neighbor_pids = GetNeighborIds(vtkdata, npid, labels, label, pid_visited)
		neighbor_pids.extend(next_neighbor_pids)
		all_connected_pids = np.append(all_connected_pids, next_neighbor_pids)

	return np.unique(all_connected_pids)

def NeighborLabel(vtkdata, labels, label, connected_pids):
	neighbor_ids = []
	
	for pid in connected_pids:
		cells_id = vtk.vtkIdList()
		vtkdata.GetPointCells(int(pid), cells_id)
		for ci in range(cells_id.GetNumberOfIds()):
			points_id_inner = vtk.vtkIdList()
			vtkdata.GetCellPoints(cells_id.GetId(ci), points_id_inner)
			for pi in range(points_id_inner.GetNumberOfIds()):
				pid_inner = points_id_inner.GetId(pi)
				if labels.GetTuple(pid_inner)[0] != label:
					neighbor_ids.append(pid_inner)

	neighbor_ids = np.unique(neighbor_ids)
	neighbor_labels = []

	for nid in neighbor_ids:
		neighbor_labels.append(labels.GetTuple(nid)[0])
	
	if len(neighbor_labels) > 0:
		return max(neighbor_labels, key=neighbor_labels.count)
	return -1



def RemoveIslands(vtkdata, labels, label, min_count,ignore_neg1 = False):

	pid_visited = np.zeros(labels.GetNumberOfTuples())
	for pid in range(labels.GetNumberOfTuples()):
		if labels.GetTuple(pid)[0] == label and pid_visited[pid] == 0:
			connected_pids = ConnectedRegion(vtkdata, pid, labels, label, pid_visited)
			if connected_pids.shape[0] < min_count:
				neighbor_label = NeighborLabel(vtkdata, labels, label, connected_pids)
				if ignore_neg1 == True and neighbor_label != -1:
					for cpid in connected_pids:
						labels.SetTuple(int(cpid), (neighbor_label,))


def ErodeLabel(vtkdata, labels, label, ignore_label=None,iterations=math.inf, target=None ):
	
	pid_labels = []
	for pid in range(labels.GetNumberOfTuples()):
		if labels.GetTuple(pid)[0] == label:
			pid_labels.append(pid)

	while pid_labels and iterations !=0:
		pid_labels_remain = pid_labels
		pid_labels = []

		all_neighbor_pids = []
		all_neighbor_labels = []

		while pid_labels_remain:

			pid = pid_labels_remain.pop()

			neighbor_pids = GetNeighbors(vtkdata, pid)
			is_neighbor = False

			for npid in neighbor_pids:
				neighbor_label = labels.GetTuple(npid)[0]
				if neighbor_label != label and (ignore_label == None or neighbor_label != ignore_label) and (target == None or neighbor_label == target):

					all_neighbor_pids.append(pid)
					all_neighbor_labels.append(neighbor_label)
					is_neighbor = True
					break

			if not is_neighbor:
				pid_labels.append(pid)

		if(all_neighbor_pids):
			for npid, nlabel in zip(all_neighbor_pids, all_neighbor_labels):
				labels.SetTuple(int(npid), (nlabel,))
		else:
			break
		iterations -= 1

def DilateLabel(vtkdata, labels, label, iterations=2, dilateOverTarget=False, target=None):
	
	pid_labels = []

	while iterations > 0:
		#Get all neighbors to the 'label' that have a different label
		all_neighbor_labels = []
		for pid in range(labels.GetNumberOfTuples()):
			if labels.GetTuple(pid)[0] == label:
				neighbor_pids = GetNeighbors(vtkdata, pid)
				pid_labels.append(pid)

				for npid in neighbor_pids:
					neighbor_label = labels.GetTuple(npid)[0]

					if dilateOverTarget:
						if neighbor_label == target:
							all_neighbor_labels.append(npid)
					else:
						if neighbor_label != label:
							all_neighbor_labels.append(npid)

		#Dilate them, i.e., change the value to label
		for npid in all_neighbor_labels:
			labels.SetTuple(int(npid), (label,))

		iterations -= 1


def GetNeighborIds(vtkdata, pid, labels, label, pid_visited):
    cells_id = vtk.vtkIdList()
    vtkdata.GetPointCells(pid, cells_id)
    neighbor_pids = []

    for ci in range(cells_id.GetNumberOfIds()):
        points_id_inner = vtk.vtkIdList()
        vtkdata.GetCellPoints(cells_id.GetId(ci), points_id_inner)
        for pi in range(points_id_inner.GetNumberOfIds()):
            pid_inner = points_id_inner.GetId(pi)
            if labels.GetTuple(pid_inner)[0] == label and pid_inner != pid and pid_visited[pid_inner] == 0:
                pid_visited[pid_inner] = 1
                neighbor_pids.append(pid_inner)

    return np.unique(neighbor_pids).tolist()


def GetNeighbors(vtkdata, pid):
    cells_id = vtk.vtkIdList()
    vtkdata.GetPointCells(pid, cells_id)
    neighbor_pids = []

    for ci in range(cells_id.GetNumberOfIds()):
        points_id_inner = vtk.vtkIdList()
        vtkdata.GetCellPoints(cells_id.GetId(ci), points_id_inner)
        for pi in range(points_id_inner.GetNumberOfIds()):
            pid_inner = points_id_inner.GetId(pi)
            if pid_inner != pid:
                neighbor_pids.append(pid_inner)

    return np.unique(neighbor_pids).tolist()

