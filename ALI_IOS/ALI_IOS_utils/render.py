import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer,
    MeshRasterizer, HardPhongShader, PointLights, blending
)

def GenPhongRenderer(image_size, blur_radius, faces_per_pixel, device):
    cameras = FoVPerspectiveCameras(znear=0.01, zfar=10, fov=90, device=device)# Initialize a perspective camera.

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
    )

    lights = PointLights(device=device)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    b = blending.BlendParams(background_color=(0, 0, 0))
    phong_renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights, blend_params=b)
    )

    # Mask renderer (defined separately in MaskRenderer class)
    from ALI_IOS_utils.mask_renderer import MaskRenderer
    mask_renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=MaskRenderer(device=device, cameras=cameras, lights=lights, blend_params=b)
    )

    return phong_renderer, mask_renderer
