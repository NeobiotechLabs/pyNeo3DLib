"""추론용 Brain 및 DNet / RNet."""
import logging

import torch
import torch.nn.functional as F
from monai.networks.nets.densenet import DenseNet
from torch import nn

from .resnet2p1d import generate_model

logger = logging.getLogger(__name__)


class Brain:
    def __init__(
        self,
        network_type,
        network_scales,
        device,
        in_channels,
        out_channels,
    ) -> None:
        self.device = device
        self.network_scales = network_scales
        self.networks = []
        for _ in network_scales:
            net = network_type(
                in_channels=in_channels,
                out_channels=out_channels,
            )
            net.to(self.device)
            self.networks.append(net)

    def Predict(self, dim, state):
        network = self.networks[dim]
        network.eval()
        with torch.no_grad():
            inp = torch.unsqueeze(state, 0).type(torch.float32).to(self.device)
            x = network(inp)
        return torch.argmax(x)

    def LoadModels(self, model_lst):
        for n, net in enumerate(self.networks):
            path = model_lst[self.network_scales[n]]
            logger.info("Loading model %s", path)
            net.load_state_dict(torch.load(path, map_location=self.device))


class DNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1024,
        out_channels: int = 6,
    ) -> None:
        super().__init__()
        self.featNet = DenseNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=in_channels,
            growth_rate=34,
            block_config=(6, 12, 24, 16),
        )
        self.dens = DN(
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def forward(self, x):
        x = self.featNet(x)
        x = self.dens(x)
        return x


class RNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1024,
        out_channels: int = 6,
    ) -> None:
        super().__init__()
        self.featNet = generate_model(
            model_depth=10,
            n_input_channels=1,
            n_classes=in_channels,
        )
        self.dens = DN(
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def forward(self, x):
        x = self.featNet(x)
        x = self.dens(x)
        return x


class DN(nn.Module):
    def __init__(self, in_channels, out_channels: int = 6) -> None:
        super().__init__()
        self.fc0 = nn.Linear(in_channels, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, out_channels)
        nn.init.xavier_uniform_(self.fc0.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
