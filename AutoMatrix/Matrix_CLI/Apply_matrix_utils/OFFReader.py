import vtk

class OFFReader():
	def __init__(self):
		FileName = None
		Output = None

	def SetFileName(self, fileName):
		self.FileName = fileName

	def GetOutput(self):
		return self.Output

	def Update(self):
		with open(self.FileName) as file:
			if 'OFF' != file.readline().strip():
				raise('Not a valid OFF header')

			n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])

			surf = vtk.vtkPolyData()
			points = vtk.vtkPoints()
			cells = vtk.vtkCellArray()

			for i_vert in range(n_verts):
				p = [float(s) for s in file.readline().strip().split(' ')]
				points.InsertNextPoint(p[0], p[1], p[2])

			for i_face in range(n_faces):
				
				t = [int(s) for s in file.readline().strip().split(' ')]

				if(t[0] == 1):
					vertex = vtk.vtkVertex()
					vertex.GetPointIds().SetId(0, t[1])
					cells.InsertNextCell(line)
				elif(t[0] == 2):
					line = vtk.vtkLine()
					line.GetPointIds().SetId(0, t[1])
					line.GetPointIds().SetId(1, t[2])
					cells.InsertNextCell(line)
				elif(t[0] == 3):
					triangle = vtk.vtkTriangle()
					triangle.GetPointIds().SetId(0, t[1])
					triangle.GetPointIds().SetId(1, t[2])
					triangle.GetPointIds().SetId(2, t[3])
					cells.InsertNextCell(triangle)

			surf.SetPoints(points)
			surf.SetPolys(cells)

			self.Output = surf
	