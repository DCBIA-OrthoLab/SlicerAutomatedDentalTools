# The Agent class responsible for rendering-based localization
import torch

from pytorch3d.renderer import look_at_rotation
from pytorch3d.structures import Meshes
import logging
import sys
# --- LOGGING CONFIGURATION ---
logger = logging.getLogger("ALI_IOS_agent")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    """Agent class for landmark localization using rendering with error handling."""
    
    def __init__(
        self,
        renderer,
        renderer2,
        camera_position,
        radius=1,
        verbose=True,
    ):
        """Initialize agent with error handling."""
        try:
            super(Agent, self).__init__()
            
            if renderer is None or renderer2 is None:
                logger.error("Renderers cannot be None")
                raise ValueError("Renderers are not properly initialized")
            
            self.renderer = renderer
            self.renderer2 = renderer2
            self.camera_points = torch.tensor(camera_position).type(torch.float32).to(DEVICE)
            self.scale = 0
            self.radius = radius
            self.verbose = verbose
            
            logger.debug(f"Agent initialized with radius: {radius}")
        except Exception as e:
            logger.error(f"Error initializing Agent: {e}")
            raise

    def position_agent(self, text, vert, label):
        """Position agent on mesh surface with error handling."""
        try:
            final_pos = torch.empty((0)).to(DEVICE)

            for mesh in range(len(text)):
                try:
                    if int(label) in text[mesh]:
                        index_pos_land = (text[mesh] == int(label)).nonzero(as_tuple=True)[0]
                        if len(index_pos_land) == 0:
                            logger.warning(f"No positions found for label {label} in mesh {mesh}")
                            final_pos = torch.cat((final_pos, torch.zeros((1, 3)).to(DEVICE)), dim=0)
                        else:
                            lst_pos = []
                            for index in index_pos_land:
                                lst_pos.append(vert[mesh][index])
                            position_agent = sum(lst_pos) / len(lst_pos)
                            final_pos = torch.cat((final_pos, position_agent.unsqueeze(0).to(DEVICE)), dim=0)
                    else:
                        final_pos = torch.cat((final_pos, torch.zeros((1, 3)).to(DEVICE)), dim=0)
                except Exception as e:
                    logger.error(f"Error positioning agent on mesh {mesh}: {e}")
                    final_pos = torch.cat((final_pos, torch.zeros((1, 3)).to(DEVICE)), dim=0)
            
            self.positions = final_pos
            logger.debug(f"Agent positioned with shape: {self.positions.shape}")
            return self.positions
        except Exception as e:
            logger.error(f"Error in position_agent: {e}")
            raise


    def GetView(self, meshes, rend=False):
        """Get view with error handling."""
        try:
            spc = self.positions
            img_lst = torch.empty((0)).to(DEVICE)
            seuil = 0.5

            for sp in self.camera_points:
                try:
                    sp_i = sp * self.radius
                    current_cam_pos = spc + sp_i
                    R = look_at_rotation(current_cam_pos, at=spc, device=DEVICE)
                    T = -torch.bmm(R.transpose(1, 2), current_cam_pos[:, :, None])[:, :, 0]

                    if rend:
                        renderer = self.renderer2
                        images = renderer(meshes_world=meshes.clone(), R=R, T=T.to(DEVICE))
                        y = images[:, :, :, :-1]

                        yr = torch.where(y[:, :, :, 0] > seuil, 1., 0.).unsqueeze(-1)
                        yg = torch.where(y[:, :, :, 1] > seuil, 2., 0.).unsqueeze(-1)
                        yb = torch.where(y[:, :, :, 2] > seuil, 3., 0.).unsqueeze(-1)

                        y = (yr + yg + yb).to(torch.float32)
                        y = y.permute(0, 3, 1, 2)

                    else:
                        renderer = self.renderer
                        images = self.renderer(meshes_world=meshes.clone(), R=R, T=T.to(DEVICE))
                        images = images.permute(0, 3, 1, 2)
                        images = images[:, :-1, :, :]

                        pix_to_face, zbuf, bary_coords, dists = self.renderer.rasterizer(meshes.clone())
                        zbuf = zbuf.permute(0, 3, 1, 2)
                        y = torch.cat([images, zbuf], dim=1)

                    img_lst = torch.cat((img_lst, y.unsqueeze(0)), dim=0)
                except Exception as e:
                    logger.error(f"Error rendering view: {e}")
                    raise
            
            img_batch = img_lst.permute(1, 0, 2, 3, 4)
            return img_batch
        except Exception as e:
            logger.error(f"Error in GetView: {e}")
            raise

    def get_view_rasterize(self, meshes):
        """Get rasterized view with error handling."""
        try:
            spc = self.positions
            img_lst = torch.empty((0)).to(DEVICE)
            tens_pix_to_face = torch.empty((0)).to(DEVICE)

            for sp in self.camera_points:
                try:
                    sp_i = sp * self.radius
                    current_cam_pos = spc + sp_i
                    R = look_at_rotation(current_cam_pos, at=spc, device=DEVICE)
                    T = -torch.bmm(R.transpose(1, 2), current_cam_pos[:, :, None])[:, :, 0]

                    renderer = self.renderer
                    images = renderer(meshes_world=meshes.clone(), R=R, T=T.to(DEVICE))
                    images = images.permute(0, 3, 1, 2)
                    images = images[:, :-1, :, :]
                    
                    temp = renderer.rasterizer(meshes.clone())
                    pix_to_face, zbuf = temp.pix_to_face, temp.zbuf

                    zbuf = zbuf.permute(0, 3, 1, 2)
                    y = torch.cat([images, zbuf], dim=1)

                    img_lst = torch.cat((img_lst, y.unsqueeze(0)), dim=0)
                    tens_pix_to_face = torch.cat((tens_pix_to_face, pix_to_face.unsqueeze(0)), dim=0)
                except Exception as e:
                    logger.error(f"Error in rasterization step: {e}")
                    raise
            
            img_batch = img_lst.permute(1, 0, 2, 3, 4)
            logger.debug(f"Rasterized view generated with shape: {img_batch.shape}")
            return img_batch, tens_pix_to_face
        except Exception as e:
            logger.error(f"Error in get_view_rasterize: {e}")
            raise