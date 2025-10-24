import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.densenet import DenseNet

class DN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels: int = 6,
    ) -> None:
        super(DN, self).__init__()

        self.fc0 = nn.Linear(in_channels,512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, out_channels)

        nn.init.xavier_uniform_(self.fc0.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)


    def forward(self,x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = x #F.softmax(self.fc3(x), dim=1)
        return output

class DNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1024,
        out_channels: int = 6,
    ) -> None:
        super(DNet, self).__init__()

        self.featNet = DenseNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=in_channels,
            growth_rate = 34,
            block_config = (6, 12, 24, 16),
        )

        self.dens = DN(
            in_channels = in_channels,
            out_channels = out_channels
        )

    def forward(self,x):
        x = self.featNet(x)
        x = self.dens(x)
        return x

class Brain:
    def __init__(
        self,
        network_type,
        network_scales,
        device,
        in_channels,
        out_channels,
        model_dir = "",
        model_name = "",
        run_dir = "",
        learning_rate = 1e-4,
        batch_size = 10,
        generate_tensorboard = False,
        verbose = False
    ) -> None:
        self.network_type = network_type
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.verbose = verbose
        self.generate_tensorboard = generate_tensorboard
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        networks = []
        global_epoch = []
        epoch_losses = []
        validation_metrics = []
        models_dirs = []

        writers = []
        optimizers = []
        best_metrics = []
        best_epoch = []

        self.network_scales = network_scales

        for n,scale in enumerate(network_scales):
            net = network_type(
                in_channels = in_channels,
                out_channels = out_channels,
            )
            net.to(self.device)
            networks.append(net)

            # num_param = sum(p.numel() for p in net.parameters())
            # print("Number of parameters :",num_param)
            # summary(net,(1,64,64,64))

            epoch_losses.append([0])
            validation_metrics.append([])
            best_metrics.append(0)
            global_epoch.append(0)
            best_epoch.append(0)

            if not model_dir == "":
                dir_path = os.path.join(model_dir,scale)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                models_dirs.append(dir_path)



        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizers = optimizers
        self.writers = writers

        self.networks = networks
        # self.networks = [networks[0]]
        self.epoch_losses = epoch_losses
        self.validation_metrics = validation_metrics
        self.best_metrics = best_metrics
        self.global_epoch = global_epoch
        self.best_epoch = best_epoch

        self.model_dirs = models_dirs
        self.model_name = model_name


    def ResetNet(self,n):
        net = self.network_type(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
        )
        net.to(self.device)
        self.networks[n] = net

        self.epoch_losses[n] = [0]
        self.validation_metrics[n] = []
        self.best_metrics[n] = 0
        self.global_epoch[n] = 0
        self.best_epoch[n] = 0


    def Predict(self,dim,state):
        network = self.networks[dim]
        network.eval()
        with torch.no_grad():
            input = torch.unsqueeze(state,0).type(torch.float32).to(self.device)
            x = network(input)
        return torch.argmax(x)

    def LoadModels(self,model_lst):
        for n,net in enumerate(self.networks):
            print("Loading model", model_lst[self.network_scales[n]])
            net.load_state_dict(torch.load(model_lst[self.network_scales[n]],map_location=self.device))