"""OSNet-AIN x1.0 person re-identification embeddings (512-d, L2-normalised).

Architecture adapted from Torchreid (Kaiyang Zhou, MIT licence: github.com/KaiyangZhou/deep-person-reid,
torchreid/models/osnet_ain.py); weights are Torchreid's multi-source (MSMT17 + DukeMTMC-reID + CUHK03)
OSNet-AIN checkpoint as redistributed unchanged in the Hugging Face repo LibreYOLO/LibreReID-osnet.

Re-ID models are trained to tell people apart by clothing and build across cameras and poses, which a
general image-text model like SigLIP does not do: on Castle5000 crops SigLIP scored two different people
in one frame higher than one person in two frames. Used only to flag possible duplicate people.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO, FILE = "LibreYOLO/LibreReID-osnet", "osnet_ain_x1_0.pt"
SHA256 = "34c24e98b6b70c8b62480f846fd0d581aa2fd1535bc0276aecf1f10430b731d1"
INPUT_HW = (256, 128)                         # Torchreid's default person crop size
MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
STD = np.array([0.229, 0.224, 0.225], dtype="float32")


class ConvLayer(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0, groups=1, IN=False):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.InstanceNorm2d(cout, affine=True) if IN else nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, cin, cout, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, cin, cout, stride=1, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(cout) if bn else None

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x) if self.bn is not None else x


class LightConv3x3(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False, groups=cout)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class LightConvStream(nn.Module):
    def __init__(self, cin, cout, depth):
        super().__init__()
        layers = [LightConv3x3(cin, cout)] + [LightConv3x3(cout, cout) for _ in range(depth - 1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ChannelGate(nn.Module):
    def __init__(self, cin, reduction=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(cin, cin // reduction, 1, bias=True, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(cin // reduction, cin, 1, bias=True, padding=0)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        w = self.gate_activation(self.fc2(self.relu(self.fc1(self.global_avgpool(x)))))
        return x * w


class OSBlock(nn.Module):
    def __init__(self, cin, cout, reduction=4, T=4):
        super().__init__()
        mid = cout // reduction
        self.conv1 = Conv1x1(cin, mid)
        self.conv2 = nn.ModuleList([LightConvStream(mid, mid, t) for t in range(1, T + 1)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, cout)
        self.downsample = Conv1x1Linear(cin, cout) if cin != cout else None

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = sum(self.gate(stream(x1)) for stream in self.conv2)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity)


class OSBlockINin(nn.Module):
    def __init__(self, cin, cout, reduction=4, T=4):
        super().__init__()
        mid = cout // reduction
        self.conv1 = Conv1x1(cin, mid)
        self.conv2 = nn.ModuleList([LightConvStream(mid, mid, t) for t in range(1, T + 1)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, cout, bn=False)
        self.downsample = Conv1x1Linear(cin, cout) if cin != cout else None
        self.IN = nn.InstanceNorm2d(cout, affine=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = sum(self.gate(stream(x1)) for stream in self.conv2)
        x3 = self.IN(self.conv3(x2))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity)


class OSNetAIN(nn.Module):
    def __init__(self, channels=(64, 256, 384, 512), feature_dim=512):
        super().__init__()
        blocks = [[OSBlockINin, OSBlockINin], [OSBlock, OSBlockINin], [OSBlockINin, OSBlock]]
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._layer(blocks[0], channels[0], channels[1])
        self.pool2 = nn.Sequential(Conv1x1(channels[1], channels[1]), nn.AvgPool2d(2, stride=2))
        self.conv3 = self._layer(blocks[1], channels[1], channels[2])
        self.pool3 = nn.Sequential(Conv1x1(channels[2], channels[2]), nn.AvgPool2d(2, stride=2))
        self.conv4 = self._layer(blocks[2], channels[2], channels[3])
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels[3], feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU())

    @staticmethod
    def _layer(kinds, cin, cout):
        return nn.Sequential(kinds[0](cin, cout), *[kind(cout, cout) for kind in kinds[1:]])

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.conv5(self.conv4(x))
        return self.fc(self.global_avgpool(x).flatten(1))


class ReidEmbedder:
    """Loads the pinned checkpoint (hash-checked) and embeds person crops given as PIL images."""

    def __init__(self, cache_dir=None, device="cpu"):
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(REPO, FILE, cache_dir=cache_dir))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != SHA256:
            raise ValueError(f"Unexpected re-ID weights (sha256 {digest[:12]}...)")
        self.model = OSNetAIN()
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(device)
        self.device = device

    def embed(self, crops) -> np.ndarray:
        """crops: PIL images of whole people. Returns (n, 512) L2-normalised vectors."""
        if not crops:
            return np.zeros((0, 512), dtype="float32")
        h, w = INPUT_HW
        batch = np.stack([(np.asarray(c.convert("RGB").resize((w, h)), dtype="float32") / 255.0 - MEAN) / STD
                          for c in crops]).transpose(0, 3, 1, 2)
        with torch.no_grad():
            v = self.model(torch.from_numpy(np.ascontiguousarray(batch)).to(self.device)).cpu().numpy()
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
