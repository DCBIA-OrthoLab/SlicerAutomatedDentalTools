# The Agent class responsible for rendering-based localization
import math

import torch

from pytorch3d.renderer import look_at_rotation

from ALI_IOS_utils.model import MG_AIM_OFFSET

# --- LOGGING CONFIGURATION ---
from ADTLib.logging_setup import get_logger

logger = get_logger("ALI_IOS_agent")

from ADTLib.env.cuda import preferred_device

# preferred_device rather than `cuda if is_available()`: that test answers for
# the driver and the runtime, not for the wheel. On a GPU the installed torch
# carries no kernels for it says True, every launch afterwards fails, and the
# agents reported the result as "landmark not found" -- seven times out of
# seven, with nothing naming the cause. The probe runs one real kernel and
# falls back to the CPU, slowly but visibly, when it cannot.
DEVICE = preferred_device()

class Agent:
    """Agent class for landmark localization using rendering with error handling."""
    
    def __init__(
        self,
        renderer,
        renderer2,
        camera_position,
        radius=1,
        verbose=True,
        lm_type=None,
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
            # 'MG' switches to the adaptive buccal 3-camera scheme; any other
            # value keeps the classic sphere scheme based on camera_position.
            self.lm_type = lm_type

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
                        elif self.lm_type == 'MG':
                            # Tensor mean, exactly like the MG training/prediction code:
                            # a sequential python sum rounds float32 differently and the
                            # resulting camera shift flips borderline pixels
                            position_agent = vert[mesh][index_pos_land].mean(dim=0)
                            final_pos = torch.cat((final_pos, position_agent.unsqueeze(0).to(DEVICE)), dim=0)
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
            # The arch frame only drives the MG cameras. O and C go through the
            # sphere scheme, which reads nothing of this, so skip it for them.
            if self.lm_type == 'MG':
                self.arch_tangents = self._arch_tangents(text, vert, label)
                self.buccal_normals, self.aim_points = self._local_frame(vert, final_pos, label)
            logger.debug(f"Agent positioned with shape: {self.positions.shape}")
            return self.positions
        except Exception as e:
            logger.error(f"Error in position_agent: {e}")
            raise

    def position_agent_estimated(self, vert, position, tangent, label):
        """Position the MG agent on a tooth absent from the segmentation.

        `position` and `tangent` come from a fit of the arch through the
        segmented teeth. The camera geometry is then built exactly as in
        position_agent: only the source of the tooth centre and of the arch
        direction changes.
        """
        self.positions = position.view(1, 3).to(DEVICE)
        tangent = tangent.clone().to(DEVICE)
        tangent[2] = 0.0                      # keep it horizontal
        norm = torch.norm(tangent)
        self.arch_tangents = (tangent / norm if norm > 1e-6 else tangent).view(1, 3)
        if self.lm_type == 'MG':
            self.buccal_normals, self.aim_points = self._local_frame(vert, self.positions, label)
        return self.positions

    def _arch_tangents(self, text, vert, label):
        """Direction of the dental arch at `label`, one row per mesh, horizontal, unit norm.

        Lower teeth carry consecutive universal ids along the arch (18 -> 31), so the
        neighbours of `label` are label-1 and label+1 and their centroids give the local
        tangent. At the ends of the arch one neighbour is missing, so a one-sided
        difference is used instead. A zero row means "unknown": the caller falls back to
        the old radial direction.
        """
        lab = int(label)
        tangents = []
        for mesh_idx in range(len(text)):
            ids = text[mesh_idx]

            def centroid(value):
                idx = (ids == value).nonzero(as_tuple=True)[0]
                return vert[mesh_idx][idx].mean(dim=0) if len(idx) > 0 else None

            here, before, after = centroid(lab), centroid(lab - 1), centroid(lab + 1)
            if before is not None and after is not None:
                tangent = after - before
            elif after is not None and here is not None:
                tangent = after - here
            elif before is not None and here is not None:
                tangent = here - before
            else:
                tangent = torch.zeros(3, device=DEVICE)

            tangent = tangent.clone()
            tangent[2] = 0.0                      # keep it horizontal
            norm = torch.norm(tangent)
            tangents.append(tangent / norm if norm > 1e-6 else tangent)

        return torch.stack(tangents)

    def _local_frame(self, vert, positions, label):
        """Buccal normal and camera aim point for each mesh, in unit-sphere space.

        The buccal normal is horizontal, perpendicular to the arch tangent and pointing
        away from the arch centre. The aim point is where the MG landmark is expected to
        be (MG_AIM_OFFSET, an anatomical prior measured on the training scans), so the
        cameras frame the gingival margin instead of the crown.
        """
        normals, aims = [], []
        offset = MG_AIM_OFFSET.get(str(int(label)))
        for mesh_idx in range(positions.shape[0]):
            tangent = self.arch_tangents[mesh_idx]
            centre = vert[mesh_idx].mean(dim=0)
            outward = positions[mesh_idx] - centre
            outward = outward.clone()
            outward[2] = 0.0

            normal = torch.stack([-tangent[1], tangent[0], torch.zeros_like(tangent[0])])
            if torch.norm(normal) < 1e-6:                     # tangent unknown
                normal = outward
            if torch.dot(normal, outward) < 0:                # face the cheek, not the tongue
                normal = -normal
            normal = normal / (torch.norm(normal) + 1e-6)

            aim = positions[mesh_idx].clone()
            if offset is not None and torch.norm(self.arch_tangents[mesh_idx]) > 1e-6:
                b, t, v = offset
                aim = aim + normal * b + tangent * t
                aim[2] = aim[2] + v
            else:                                             # old behaviour as a fallback
                aim[2] = aim[2] - 0.2
            normals.append(normal)
            aims.append(aim)
        return torch.stack(normals), torch.stack(aims)


    def _mg_camera_directions(self, spc, meshes):
        """
        Compute the 3 buccal camera directions (front + left/right rotated by +/-0.35 rad)
        used by the mucogingival (MG) model. The vertical axis is Z, so it is zeroed to
        keep the directions horizontal. Must stay identical to the training code,
        otherwise the model inputs no longer match what it was trained on.
        Returns directions of shape [B, 3, 3].

        The buccal direction is the horizontal normal to the arch at this tooth, i.e.
        perpendicular to the local arch tangent and pointing away from the arch centre.
        The radial direction (tooth centre - mesh centre) used before is only buccal near
        the midline: measured against the true normal it is off by 2-5 deg on the incisors
        but 35 deg on tooth 19/30 and 53 deg on tooth 31, so on the molars the cameras
        looked ALONG the arch and the landmark fell outside the render entirely.
        """
        center = meshes.verts_padded().mean(dim=1, keepdim=True)
        hauteur_idx = 2   # vertical axis (Z)
        plane_idx = 1     # horizontal rotation axis (Y)

        outward = spc - center
        outward[:, :, hauteur_idx] = 0
        outward = outward / (torch.norm(outward, dim=-1, keepdim=True) + 1e-6)

        normals = getattr(self, "buccal_normals", None)
        if normals is not None and normals.shape[0] >= spc.shape[0]:
            direction = normals[:spc.shape[0]].unsqueeze(1).to(spc.device)   # [B, 1, 3]
        else:
            direction = outward

        angle = 0.35
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        dir_front = direction
        dir_left = dir_front.clone()
        dir_right = dir_front.clone()
        dir_left[:, :, 0] = dir_front[:, :, 0] * cos_a - dir_front[:, :, plane_idx] * sin_a
        dir_left[:, :, plane_idx] = dir_front[:, :, 0] * sin_a + dir_front[:, :, plane_idx] * cos_a
        dir_right[:, :, 0] = dir_front[:, :, 0] * cos_a + dir_front[:, :, plane_idx] * sin_a
        dir_right[:, :, plane_idx] = -dir_front[:, :, 0] * sin_a + dir_front[:, :, plane_idx] * cos_a
        return torch.cat([dir_front, dir_left, dir_right], dim=1)

    def _mg_camera_RT(self, spc, directions):
        """
        Build the look-at rotation R and translation T for the MG camera directions.
        Cameras aim at the expected landmark position (self.aim_points) and sit exactly
        self.radius away from it, so every tooth is framed at the same scale and the
        gingival margin lands near the centre of the image.

        Aiming at the tooth centre lowered by a flat 0.2, as before, only framed the
        incisors: the molar landmark is ~0.15 further buccal and projected outside the
        224 px image (measured at 300-455 px), which is why its target was empty.
        Returns R [B*K, 3, 3] and T [B*K, 3].
        """
        hauteur_idx = 2
        n_cameras = directions.shape[1]

        aims = getattr(self, "aim_points", None)
        if aims is not None and aims.shape[0] >= spc.shape[0]:
            centre = aims[:spc.shape[0]].unsqueeze(1).to(spc.device)   # [B, 1, 3]
        else:
            centre = spc.clone()
            centre[:, :, hauteur_idx] -= 0.2

        cam_pos = centre + directions * self.radius
        cam_pos[:, :, hauteur_idx] -= (self.radius * 0.15)      # keep the slight plunge

        target = centre.expand(-1, n_cameras, -1).clone()

        cam_flat = cam_pos.reshape(-1, 3)
        target_flat = target.reshape(-1, 3)
        up = torch.tensor([0.0, 0.0, 1.0], device=DEVICE).view(1, 3).expand(cam_flat.shape[0], -1)

        R = look_at_rotation(cam_flat, at=target_flat, up=up, device=DEVICE)
        T = -torch.bmm(R.transpose(1, 2), cam_flat[:, :, None])[:, :, 0]
        return R, T

    def _get_view_rasterize_mg(self, meshes):
        """
        MG rendering used at prediction time. Renders and rasterizes each of the 3
        adaptive buccal cameras with the SAME R,T so pix_to_face stays aligned with
        the rendered images. Returns images [B, 3, 4, H, W] (RGB + Z per camera) and
        pix_to_face [B, 3, H, W, 1].
        """
        spc = self.positions.view(-1, 1, 3)[:len(meshes)]
        directions = self._mg_camera_directions(spc, meshes)
        r_all, t_all = self._mg_camera_RT(spc, directions)

        img_lst = []
        tens_pix_to_face = []
        for cam_idx in range(directions.shape[1]):
            R = r_all[cam_idx:cam_idx + 1]
            T = t_all[cam_idx:cam_idx + 1]

            images = self.renderer(meshes_world=meshes, R=R, T=T.to(DEVICE)).permute(0, 3, 1, 2)
            rgb = images[:, :-1, :, :]

            fragments = self.renderer.rasterizer(meshes, R=R, T=T.to(DEVICE))
            pix_to_face = fragments.pix_to_face
            zbuf = fragments.zbuf.permute(0, 3, 1, 2)

            y = torch.cat([rgb, zbuf], dim=1)
            img_lst.append(y.unsqueeze(1))
            tens_pix_to_face.append(pix_to_face.unsqueeze(1))

        return torch.cat(img_lst, dim=1), torch.cat(tens_pix_to_face, dim=1)

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
            if self.lm_type == 'MG':
                return self._get_view_rasterize_mg(meshes)

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