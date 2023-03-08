import vtk 
import os



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
	



def ReadSurf(path):

    fname, extension = os.path.splitext(os.path.basename(path))
    extension = extension.lower()
    if extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        surf = reader.GetOutput()    
    elif extension == ".stl":
        reader = vtk.vtkSTLReader()
        reader.SetFileName(path)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".off":
        reader = OFFReader()
        reader.SetFileName(path)
        reader.Update()
        surf = reader.GetOutput()
    elif extension == ".obj":
        if os.path.exists(fname + ".mtl"):
            obj_import = vtk.vtkOBJImporter()
            obj_import.SetFileName(path)
            obj_import.SetFileNameMTL(fname + ".mtl")
            textures_path = os.path.normpath(os.path.dirname(fname) + "/../images")
            if os.path.exists(textures_path):
                textures_path = os.path.normpath(fname.replace(os.path.basename(fname), ''))
                obj_import.SetTexturePath(textures_path)
            else:
                textures_path = os.path.normpath(fname.replace(os.path.basename(fname), ''))                
                obj_import.SetTexturePath(textures_path)
                    

            obj_import.Read()

            actors = obj_import.GetRenderer().GetActors()
            actors.InitTraversal()
            append = vtk.vtkAppendPolyData()

            for i in range(actors.GetNumberOfItems()):
                surfActor = actors.GetNextActor()
                append.AddInputData(surfActor.GetMapper().GetInputAsDataSet())
            
            append.Update()
            surf = append.GetOutput()
            
        else:
            reader = vtk.vtkOBJReader()
            reader.SetFileName(path)
            reader.Update()
            surf = reader.GetOutput()

    return surf


def WriteSurf(surf, output_folder,name):
    dir, name = os.path.split(name)
    name, extension = os.path.splitext(name)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(os.path.join(output_folder,f"{name}.vtk"))
    writer.SetInputData(surf)
    writer.Update()
