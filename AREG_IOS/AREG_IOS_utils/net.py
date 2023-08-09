import torch
import monai

# import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torchmetrics
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_rotation,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    AmbientLights,
    TexturesVertex,
)
from pytorch3d.structures import Meshes


class TimeDistributed(torch.nn.Module):
    def __init__(self, module, prediction=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.prediction = prediction

    def forward(self, input_seq):
        assert len(input_seq.size()) > 2

        size = input_seq.size()

        batch_size = size[0]
        time_steps = size[1]

        size_reshape = [batch_size * time_steps] + list(size[2:])
        reshaped_input = input_seq.contiguous().view(size_reshape)

        output = self.module(reshaped_input)

        output_size = output.size()
        output_size = [batch_size, time_steps] + list(output_size[1:])
        output = output.contiguous().view(output_size)

        return output


class MonaiUNetHRes(LightningModule):
    """
    MonaiUnetHRes class allows rendering the mesh and predict the patch
    """

    def __init__(
        self,
        args=None,
        out_channels=2,
        in_channels=4,
        class_weights=None,
        image_size=320,
    ):

        super(MonaiUNetHRes, self).__init__()

        self.save_hyperparameters()
        self.args = args
        self.image_size = image_size

        self.out_channels = out_channels
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights).to(torch.float32)

        self.loss = monai.losses.DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            ce_weight=self.class_weights,
        )
        self.accuracy = torchmetrics.Accuracy(
            num_classes=out_channels, task="multiclass"
        )

        unet = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        self.model = TimeDistributed(unet)

        self.setup_ico_verts()
        self.renderer = self.setup_render()

    def setup_ico_verts(self):

        self.ico_verts = torch.tensor(
            [
                [0, 0, 0.9],
                [0.2, 0, 0.9],
                [-0.2, 0, 0.9],
                [0, 0.2, 0.9],
                [0, -0.2, 0.9],
                [-0.2, -0.2, 0.9],
                [0.2, -0.2, 0.9],
            ],
            device=self.device,
        ).to(
            torch.float32
        )  # position of the camera
        self.number_image = self.ico_verts.shape[0]

    def setup_render(self):
        cameras = FoVPerspectiveCameras()
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0,
            faces_per_pixel=1,
            max_faces_per_bin=200000,
            perspective_correct=True,
        )

        lights = AmbientLights()

        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        phong_renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=HardPhongShader(cameras=cameras, lights=lights),
        )

        return phong_renderer

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        return optimizer

    def to(self, device=None):
        self.renderer = self.renderer.to(device)
        return super().to(device)

    # call this function to predict the patch
    def forward(self, x):

        V, F, CN = x

        V = V.to(self.device, non_blocking=True)  # mean vertex (1,number of vertex,3)
        F = F.to(self.device, non_blocking=True)  # mean faces (1,number of faces , 3)
        CN = CN.to(self.device, non_blocking=True).to(
            torch.float32
        )  # mean color normal (1, number of vertex , 3)

        X, PF = self.render(V, F, CN)
        # PF mean pixel-to-faces, it s link between the pixel in image X and the faces of a mesh,
        # X.shape->(1,7,4,320,320) PF.shape->(1,7,1,320,320)
        # 7 because 7 cameras, 4 because 4 channels :3 for rgb (normals) and 1 for the depth map
        # 320 dimension of the picture

        x = self.model(
            X
        )  # x is prediction of the neural network (1,7,2,320,320) 2 because 2 out channels

        return x, X, PF

    def render(self, V, F, CN):

        textures_normal = TexturesVertex(verts_features=CN)
        meshes = Meshes(verts=V, faces=F, textures=textures_normal)

        PF = []
        X = []
        for camera_position in self.ico_verts:

            camera_position = camera_position.unsqueeze(0).to(self.device)

            R = look_at_rotation(camera_position, device=self.device)  # (1, 3, 3)
            T = -torch.bmm(R.transpose(1, 2), camera_position[:, :, None])[
                :, :, 0
            ]  # (1, 3)

            images = self.renderer(meshes_world=meshes.clone(), R=R, T=T)
            fragments = self.renderer.rasterizer(meshes.clone())
            zbuf = fragments.zbuf  # depth map
            pf = fragments.pix_to_face

            images = torch.cat([images[:, :, :, 0:3], zbuf], dim=-1)
            images = images.permute(0, 3, 1, 2)

            pf = pf.permute(0, 3, 1, 2)

            PF.append(pf.unsqueeze(1))
            X.append(images.unsqueeze(1))

        X = torch.cat(X, dim=1)
        PF = torch.cat(PF, dim=1)

        return X, PF

    def training_step(self, train_batch, batch_idx):

        V, F, CN, LF = train_batch

        V = V.to(self.device, non_blocking=True)
        F = F.to(self.device, non_blocking=True)
        CN = CN.to(self.device, non_blocking=True).to(torch.float32)
        LF = LF.to(self.device, non_blocking=True)

        x, X, PF = self((V, F, CN))

        y = torch.take(LF, PF) * (PF >= 0)

        x = x.permute(0, 2, 1, 3, 4)
        y = y.permute(0, 2, 1, 3, 4)

        loss = self.loss(x, y)

        batch_size = V.shape[0]

        self.log("train_loss", loss, batch_size=batch_size)
        self.accuracy(
            torch.argmax(x, dim=1, keepdim=True).reshape(-1, 1),
            y.reshape(-1, 1).to(torch.int32),
        )
        self.log("train_acc", self.accuracy, batch_size=batch_size)

        return loss

    def validation_step(self, val_batch, batch_idx):
        V, F, CN, LF = val_batch

        V = V.to(self.device, non_blocking=True)
        F = F.to(self.device, non_blocking=True)
        CN = CN.to(self.device, non_blocking=True).to(torch.float32)
        LF = LF.to(self.device, non_blocking=True)

        x, X, PF = self((V, F, CN))

        y = torch.take(LF, PF) * (PF >= 0)

        x = x.permute(0, 2, 1, 3, 4)
        y = y.permute(0, 2, 1, 3, 4)
        loss = self.loss(x, y)

        batch_size = V.shape[0]
        self.accuracy(
            torch.argmax(x, dim=1, keepdim=True).reshape(-1, 1),
            y.reshape(-1, 1).to(torch.int32),
        )
        self.log("val_acc", self.accuracy, batch_size=batch_size, sync_dist=True)
        self.log("val_loss", loss, batch_size=batch_size, sync_dist=True)

    def test_step(self, test_batch, batch_idx):

        V, F, CN, LF = test_batch

        V = V.to(self.device, non_blocking=True)
        F = F.to(self.device, non_blocking=True)
        CN = CN.to(self.device, non_blocking=True).to(torch.float32)
        LF = LF.to(self.device, non_blocking=True)

        x, X, PF = self((V, F, CN))

        y = torch.take(LF, PF) * (PF >= 0)

        x = x.permute(0, 2, 1, 3, 4)
        y = y.permute(0, 2, 1, 3, 4)
        loss = self.loss(x, y)

        self.accuracy(
            torch.argmax(x, dim=1, keepdim=True).reshape(-1, 1),
            y.reshape(-1, 1).to(torch.int32),
        )

        return {"test_loss": loss, "test_correct": self.accuracy}
