# The Agent class responsible for rendering-based localization
import torch
from pytorch3d.renderer import look_at_rotation
from pytorch3d.structures import Meshes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(
        self,
        renderer,
        renderer2,
        camera_position,
        radius = 1,
        verbose = True,
        ):
        super(Agent, self).__init__()
        self.renderer = renderer
        self.renderer2=renderer2
        self.camera_points = torch.tensor(camera_position).type(torch.float32).to(DEVICE)
        self.scale = 0
        self.radius = radius
        self.verbose = verbose


    def position_agent(self, text, vert, label):

        final_pos = torch.empty((0)).to(DEVICE)

        for mesh in range(len(text)):
            if int(label) in text[mesh]:
                index_pos_land = (text[mesh]==int(label)).nonzero(as_tuple=True)[0]
                lst_pos = []
                for index in index_pos_land:
                    lst_pos.append(vert[mesh][index])
                position_agent = sum(lst_pos)/len(lst_pos)
                final_pos = torch.cat((final_pos,position_agent.unsqueeze(0).to(DEVICE)),dim=0)
            else:
                final_pos = torch.cat((final_pos,torch.zeros((1,3)).to(DEVICE)),dim=0)
        # print(final_pos.shape)
        self.positions = final_pos
        # print(self.positions)
        return self.positions


    def GetView(self,meshes,rend=False):
        spc = self.positions
        img_lst = torch.empty((0)).to(DEVICE)
        seuil = 0.5

        for sp in self.camera_points:
            sp_i = sp*self.radius
            # sp = sp.unsqueeze(0).repeat(self.batch_size,1)
            current_cam_pos = spc + sp_i
            R = look_at_rotation(current_cam_pos, at=spc, device=DEVICE)  # (1, 3, 3)
            # print( 'R shape :',R.shape)
            # print(R)
            T = -torch.bmm(R.transpose(1, 2), current_cam_pos[:, :, None])[:, :, 0]  # (1, 3)

            if rend:
                renderer = self.renderer2
                images = renderer(meshes_world=meshes.clone(), R=R, T=T.to(DEVICE))
                y = images[:,:,:,:-1]

                # yd = torch.where(y[:,:,:,:]<=seuil,0.,0.)
                yr = torch.where(y[:,:,:,0]>seuil,1.,0.).unsqueeze(-1)
                yg = torch.where(y[:,:,:,1]>seuil,2.,0.).unsqueeze(-1)
                yb = torch.where(y[:,:,:,2]>seuil,3.,0.).unsqueeze(-1)

                y = ( yr + yg + yb).to(torch.float32)

                y = y.permute(0,3,1,2)

            else:
                renderer = self.renderer
                images = self.renderer(meshes_world=meshes.clone(), R=R, T=T.to(DEVICE))
                images = images.permute(0,3,1,2)
                images = images[:,:-1,:,:]

                pix_to_face, zbuf, bary_coords, dists = self.renderer.rasterizer(meshes.clone())
                zbuf = zbuf.permute(0, 3, 1, 2)
                y = torch.cat([images, zbuf], dim=1)

            img_lst = torch.cat((img_lst,y.unsqueeze(0)),dim=0)
        img_batch =  img_lst.permute(1,0,2,3,4)

        return img_batch

    def get_view_rasterize(self,meshes):
        spc = self.positions
        img_lst = torch.empty((0)).to(DEVICE)
        tens_pix_to_face = torch.empty((0)).to(DEVICE)

        for sp in self.camera_points:
            sp_i = sp*self.radius
            current_cam_pos = spc + sp_i
            R = look_at_rotation(current_cam_pos, at=spc, device=DEVICE)  # (1, 3, 3)
            T = -torch.bmm(R.transpose(1, 2), current_cam_pos[:, :, None])[:, :, 0]  # (1, 3)

            renderer = self.renderer
            images = renderer(meshes_world=meshes.clone(), R=R, T=T.to(DEVICE))
            images = images.permute(0,3,1,2)
            images = images[:,:-1,:,:]
            #pix_to_face, zbuf, bary_coords, dists = renderer.rasterizer(meshes.clone())
            temp = renderer.rasterizer(meshes.clone())
            pix_to_face, zbuf = temp.pix_to_face, temp.zbuf

            '''< Class : pytorch3d.renderer.mesh.rasterizer.Fragments >'''
            '''TypeError: cannot unpack non-iterable Fragments object'''
            zbuf = zbuf.permute(0, 3, 1, 2)
            y = torch.cat([images, zbuf], dim=1)

            img_lst = torch.cat((img_lst,y.unsqueeze(0)),dim=0)
            tens_pix_to_face = torch.cat((tens_pix_to_face,pix_to_face.unsqueeze(0)),dim=0)
        img_batch =  img_lst.permute(1,0,2,3,4)

        return img_batch , tens_pix_to_face