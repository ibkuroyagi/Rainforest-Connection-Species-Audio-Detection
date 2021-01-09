# -*- coding: utf-8 -*-

# Created by Ibuki Kuroyanagi

"""Conformer algorithm."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.augmentation import SpecAugmentation
from efficientnet_pytorch import EfficientNet

from .cnn import AttBlock
from .utils import init_layer


class EfficientNet_b(nn.Module):
    def __init__(
        self,
        classes_num=24,
        efficient_net_name="efficientnet-b0",
        feat_dim=1280,
        training=False,
        is_spec_augmenter=False,
    ):

        super(self.__class__, self).__init__()
        self.conv0 = nn.Conv2d(1, 3, 1, 1)
        self.efficientnet = EfficientNet.from_pretrained(efficient_net_name)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(feat_dim, feat_dim, bias=True)
        self.dropout2 = nn.Dropout(p=0.2)
        self.att_block = AttBlock(feat_dim, classes_num, activation="linear")

        self.init_weight()
        self.training = training
        self.is_spec_augmenter = is_spec_augmenter
        if is_spec_augmenter:
            # Spec augmenter
            self.spec_augmenter = SpecAugmentation(
                time_drop_width=80,
                time_stripes_num=2,
                freq_drop_width=20,
                freq_stripes_num=2,
            )

    def init_weight(self):
        init_layer(self.fc1)

    def forward(self, input):
        """Input: (batch_size, mels, T')"""

        x = input.unsqueeze(3)
        x = x.transpose(1, 3)  # (B, 1, T', mels)
        if self.training and self.is_spec_augmenter:
            x = self.spec_augmenter(x)
        x = self.conv0(x)
        x = self.efficientnet.extract_features(x)
        # print(f"feature_map:{x.shape}")
        x = torch.mean(x, dim=3)  # + torch.max(x, dim=3)[0]
        embedding = torch.mean(x, dim=2)
        # print(f"feature_map: mean-dim3{x.shape}")
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(self.dropout1(x)))
        x = x.transpose(1, 2)
        # print(f"pool1d_map: mean-dim3{x.shape}")
        (clipwise_output, _, segmentwise_output) = self.att_block(self.dropout2(x))
        segmentwise_output = segmentwise_output.transpose(1, 2)

        output_dict = {
            "y_frame": segmentwise_output,  # (B, T', n_class)
            "y_clip": clipwise_output,  # (B, n_class)
            "embedding": embedding,  # (B, feat_dim)
        }

        return output_dict


class EfficientNet_simple(nn.Module):
    def __init__(
        self,
        classes_num=24,
        efficient_net_name="efficientnet-b0",
        feat_dim=1280,
        training=False,
        is_spec_augmenter=False,
    ):

        super(self.__class__, self).__init__()
        self.conv0 = nn.Conv2d(1, 3, 1, 1)
        self.efficientnet = EfficientNet.from_pretrained(efficient_net_name)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(feat_dim, feat_dim, bias=True)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(feat_dim, classes_num, bias=True)

        self.init_weight()
        self.training = training
        self.is_spec_augmenter = is_spec_augmenter
        if is_spec_augmenter:
            # Spec augmenter
            self.spec_augmenter = SpecAugmentation(
                time_drop_width=80,
                time_stripes_num=2,
                freq_drop_width=20,
                freq_stripes_num=2,
            )

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, input):
        """Input: (batch_size, mels, T')"""

        x = input.unsqueeze(3)
        x = x.transpose(1, 3)  # (B, 1, T', mels)
        if self.training and self.is_spec_augmenter:
            x = self.spec_augmenter(x)
        x = self.conv0(x)
        x = self.efficientnet.extract_features(x)
        # print(f"feature_map:{x.shape}")
        x = torch.mean(x, dim=3) + torch.max(x, dim=3)[0]
        embedding = torch.mean(x, dim=2)
        # print(f"feature_map: mean-dim3{x.shape}")
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(self.dropout1(x)))
        segmentwise_output = self.fc2(self.dropout2(x))
        # print(f"segmentwise_output{segmentwise_output.shape}")
        clipwise_output = segmentwise_output.max(dim=1)[0]

        output_dict = {
            "y_frame": segmentwise_output,  # (B, T', n_class)
            "y_clip": clipwise_output,  # (B, n_class)
            "embedding": embedding,  # (B, feat_dim)
        }

        return output_dict
