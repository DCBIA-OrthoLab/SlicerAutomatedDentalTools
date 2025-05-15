import torch.nn as nn
from pytorch3d.renderer import Materials, BlendParams, PointLights
from pytorch3d.renderer.blending import hard_rgb_blend
from pytorch3d.renderer.utils import TensorProperties
from typing import Optional
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.rasterizer import Fragments

class MaskRenderer(nn.Module):
    def __init__(self, device="cpu",
                 cameras: Optional[TensorProperties] = None,
                 lights: Optional[TensorProperties] = None,
                 materials: Optional[Materials] = None,
                 blend_params: Optional[BlendParams] = None):
        super().__init__()
        self.lights = lights if lights else PointLights(device=device)
        self.materials = materials if materials else Materials(device=device)
        self.cameras = cameras
        self.blend_params = blend_params if blend_params else BlendParams()

    def to(self, device):
        if self.cameras:
            self.cameras = self.cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs):
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            raise ValueError("Cameras must be provided in the forward pass or during init.")
        texels = meshes.sample_textures(fragments)
        blend_params = kwargs.get("blend_params", self.blend_params)
        return hard_rgb_blend(texels, fragments, blend_params)
