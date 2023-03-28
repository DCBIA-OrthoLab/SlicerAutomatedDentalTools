import torch.nn as nn
import torch.optim as optim
from slicer.util import pip_install, pip_uninstall

try:
    import pytorch_lightning as pl
except ImportError:
    pip_install('pytorch_lightning')
    import pytorch_lightning as pl
    
pip_uninstall('monai')
pip_install('monai')
from monai.networks.nets.densenet import DenseNet169

# Different Network

class DenseNet(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.lr = lr
        self.net = DenseNet169(spatial_dims=3, in_channels=1, out_channels=3)
        self.CosSimLoss = nn.CosineSimilarity()

    def forward(self, x):
        return nn.functional.normalize(self.net(x),dim=1)

    def training_step(self, batch, batch_idx):
        scan, directionVector, scan_path = batch
        batch_size = scan.shape[0]

        directionVector_hat = self(scan)
        
        loss = (1 - self.CosSimLoss(directionVector_hat, directionVector))
        # Sum the loss over the batch
        loss = loss.sum()
        self.log('train_loss', loss, batch_size=batch_size)
                   
        return loss

    def validation_step(self, batch, batch_idx):
        scan, directionVector, scan_path = batch
        batch_size = scan.shape[0]
        directionVector_hat = self(scan)
        
        loss = (1 - self.CosSimLoss(directionVector_hat, directionVector))
        loss = loss.sum()
        self.log('val_loss', loss, batch_size=batch_size)

        return loss

    def test_step(self, batch, batch_idx):
        scan, directionVector, scan_path = batch
        batch_size = scan.shape[0]

        directionVector_hat = self(scan)
        
        loss = (1 - self.CosSimLoss(directionVector_hat, directionVector))
        loss = loss.sum()
        self.log('test_loss', loss, batch_size=batch_size)
        
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)