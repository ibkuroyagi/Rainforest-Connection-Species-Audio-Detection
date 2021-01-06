import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchlibrosa.augmentation import SpecAugmentation
from .cnn import AttBlock
from .utils import init_layer


class MobileNetV2(nn.Module):
    def __init__(
        self,
        classes_num=24,
        feat_dim=1280,
        training=False,
        is_spec_augmenter=False,
    ):

        super(self.__class__, self).__init__()
        self.conv0 = nn.Conv2d(1, 3, 1, 1)
        self.mobilenetv2 = torchvision.models.mobilenet_v2(pretrained=True)
        self.fc1 = nn.Linear(feat_dim, feat_dim, bias=True)
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
        x = self.mobilenetv2.features(x)
        # print(f"feature_map:{x.shape}")
        x = torch.mean(x, dim=3) + torch.max(x, dim=3)[0]
        embedding = torch.mean(x, dim=2)
        # print(f"feature_map: mean-dim3{x.shape}")
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        # print(f"pool1d_map: mean-dim3{x.shape}")
        (clipwise_output, _, segmentwise_output) = self.att_block(x)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        output_dict = {
            "y_frame": segmentwise_output,  # (B, T', n_class)
            "y_clip": clipwise_output,  # (B, n_class)
            "embedding": embedding,  # (B, feat_dim)
        }

        return output_dict


class MobileNetV2_simple(nn.Module):
    def __init__(
        self,
        classes_num=24,
        feat_dim=1280,
        training=False,
        is_spec_augmenter=False,
    ):

        super(self.__class__, self).__init__()
        self.conv0 = nn.Conv2d(1, 3, 1, 1)
        self.mobilenetv2 = torchvision.models.mobilenet_v2(pretrained=True)
        self.fc1 = nn.Linear(feat_dim, feat_dim, bias=True)
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
        x = self.mobilenetv2.features(x)
        print(f"feature_map:{x.shape}")
        x = torch.mean(x, dim=3) + torch.max(x, dim=3)[0]
        embedding = torch.mean(x, dim=2)
        print(f"feature_map: mean-dim3{x.shape}")
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        segmentwise_output = self.fc2(x)
        print(f"segmentwise_output{segmentwise_output.shape}")
        clipwise_output = segmentwise_output.max(dim=1)[0]

        output_dict = {
            "y_frame": segmentwise_output,  # (B, T', n_class)
            "y_clip": clipwise_output,  # (B, n_class)
            "embedding": embedding,  # (B, feat_dim)
        }

        return output_dict
