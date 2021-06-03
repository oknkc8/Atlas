# from Detectron2: (https://github.com/facebookresearch/detectron2)

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from detectron2.layers import Conv2d, get_norm
from detectron2.modeling.backbone import build_backbone as d2_build_backbone
import fvcore.nn.weight_init as weight_init



def build_backbone2d(cfg):
    """ Builds 2D feature extractor backbone network from Detectron2."""

    output_dim = cfg.MODEL.BACKBONE3D.CHANNELS[0] - 1
    norm = cfg.MODEL.FPN.NORM
    output_stride = 4  # TODO: make configurable

    backbone = d2_build_backbone(cfg)
    feature_extractor = FPNFeature(
        backbone.output_shape(), output_dim, output_stride, norm)

    # load pretrained backbone
    if cfg.MODEL.BACKBONE.WEIGHTS:
        state_dict = torch.load(cfg.MODEL.BACKBONE.WEIGHTS)
        backbone.load_state_dict(state_dict)

    return nn.Sequential(backbone, feature_extractor), output_stride


class FPNFeature(nn.Module):
    """ Converts feature pyrimid to singe feature map (from Detectron2)"""
    
    def __init__(self, input_shape, output_dim=32, output_stride=4, norm='BN'):
        super().__init__()

        # fmt: off
        self.in_features      = ["p2", "p3", "p4", "p5"]
        feature_strides       = {k: v.stride for k, v in input_shape.items()}
        feature_channels      = {k: v.channels for k, v in input_shape.items()}
        # fmt: on

        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(output_stride))
            )
            for k in range(head_length):
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else output_dim,
                    output_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=get_norm(norm, output_dim),
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != output_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])

    def forward(self, features):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])
        return x


############################################################
#        MNasNet for Lightweight Feature Extraction        #
############################################################


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding upimport torchvision
    rather than down. """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


class MnasMulti(nn.Module):

    def __init__(self, alpha=1.0):
        super(MnasMulti, self).__init__()
        depths = _get_depths(alpha)
        if alpha == 1.0:
            MNASNet = torchvision.models.mnasnet1_0(pretrained=True, progress=True)
        else:
            MNASNet = torchvision.models.MNASNet(alpha=alpha)

        self.conv0 = nn.Sequential(
            MNASNet.layers._modules['0'],
            MNASNet.layers._modules['1'],
            MNASNet.layers._modules['2'],
            MNASNet.layers._modules['3'],
            MNASNet.layers._modules['4'],
            MNASNet.layers._modules['5'],
            MNASNet.layers._modules['6'],
            MNASNet.layers._modules['7'],
            MNASNet.layers._modules['8'],
        )

        self.conv1 = MNASNet.layers._modules['9']
        self.conv2 = MNASNet.layers._modules['10']

        #self.out1 = nn.Conv2d(depths[4], depths[4], 1, bias=False)
        #self.out_channels = [depths[4]]

        final_chs = depths[4]
        self.inner1 = nn.Conv2d(depths[3], final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(depths[2], final_chs, 1, bias=True)

        #self.out2 = nn.Conv2d(final_chs, depths[3], 3, padding=1, bias=False)
        #self.out3 = nn.Conv2d(final_chs, depths[2], 3, padding=1, bias=False)
        #self.out_channels.append(depths[3])
        #self.out_channels.append(depths[2])

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        #outputs = []
        #out = self.out1(intra_feat)
        #outputs.append(out)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
        #out = self.out2(intra_feat)
        #outputs.append(out)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
        out = self.out3(intra_feat)
        #outputs.append(out)

        #return outputs[::-1]
        return out
