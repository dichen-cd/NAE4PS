'''
Minor modificatin from https://github.com/pytorch/vision/blob/2b73a4846773a670632b29fb2fc2ac57df7bce5d/torchvision/models/detection/backbone_utils.py#L43
'''
from collections import OrderedDict

from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import BackboneWithFPN


class BackboneWithFasterRCNN(nn.Sequential):

    def __init__(self, backbone):
        super(BackboneWithFasterRCNN, self).__init__(
            OrderedDict([
                ['conv1', backbone.conv1],
                ['bn1', backbone.bn1],
                ['relu', backbone.relu],
                ['maxpool', backbone.maxpool],
                ['layer1', backbone.layer1],  # res2
                ['layer2', backbone.layer2],  # res3
                ['layer3', backbone.layer3]]  # res4
            )
        )
        self.out_channels = 1024

    def forward(self, x):
        # using the forward method from nn.Sequential
        feat = super(BackboneWithFasterRCNN, self).forward(x)
        return OrderedDict([['feat_res4', feat]])


class RCNNConvHead(nn.Sequential):
    """docstring for RCNNConvHead"""

    def __init__(self, backbone, return_res4=True, GAP=True):
        super(RCNNConvHead, self).__init__(
            OrderedDict(
                [['layer4', backbone.layer4]]  # res5
            )
        )
        self.return_res4 = return_res4
        if self.return_res4:
            self.out_channels = [1024, 2048]
        else:
            self.out_channels = [2048]
        self.GAP = GAP

    def forward(self, x):
        feat = super(RCNNConvHead, self).forward(x)
        if self.GAP:
            x = F.adaptive_max_pool2d(x, 1)
            feat = F.adaptive_max_pool2d(feat, 1)
        if self.return_res4:
            return OrderedDict([
                ['feat_res4', x],  # Global average pooling
                ['feat_res5', feat]]
            )
        else:
            return OrderedDict([['feat_res5', feat]])


def resnet_backbone(backbone_name, pretrained,
                    return_res4=True, GAP=True):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained)

    # freeze layers
    backbone.conv1.weight.requires_grad_(False)
    backbone.bn1.weight.requires_grad_(False)
    backbone.bn1.bias.requires_grad_(False)

    stem = BackboneWithFasterRCNN(backbone)
    head = RCNNConvHead(backbone, return_res4, GAP)

    return stem, head
