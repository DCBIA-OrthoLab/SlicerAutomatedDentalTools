
#try to upgrade pip






import torch.nn as nn
import torch.optim as optim


import pytorch_lightning as pl



from monai.networks.nets.densenet import DenseNet169


# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger

logger = get_logger("ASO_CBCT_Net")

# Different Network


class DenseNet(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.lr = lr
        self.net = DenseNet169(spatial_dims=3, in_channels=1, out_channels=3)
        self.CosSimLoss = nn.CosineSimilarity()

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=1)

    def training_step(self, batch, batch_idx):
        scan, direction_vector, scan_path = batch
        batch_size = scan.shape[0]

        direction_vector_hat = self(scan)

        loss = 1 - self.CosSimLoss(direction_vector_hat, direction_vector)
        # Sum the loss over the batch
        loss = loss.sum()
        self.log("train_loss", loss, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        scan, direction_vector, scan_path = batch
        batch_size = scan.shape[0]
        direction_vector_hat = self(scan)

        loss = 1 - self.CosSimLoss(direction_vector_hat, direction_vector)
        loss = loss.sum()
        self.log("val_loss", loss, batch_size=batch_size)

        return loss

    def test_step(self, batch, batch_idx):
        scan, direction_vector, scan_path = batch
        batch_size = scan.shape[0]

        direction_vector_hat = self(scan)

        loss = 1 - self.CosSimLoss(direction_vector_hat, direction_vector)
        loss = loss.sum()
        self.log("test_loss", loss, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
