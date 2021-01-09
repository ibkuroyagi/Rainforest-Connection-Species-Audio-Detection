# -*- coding: utf-8 -*-

# Copyright 2020 Ibuki Kuroyanagi
#  MIT License (https://opensource.org/licenses/MIT)

"""Collater function modules."""
import sys
import random
import logging
import numpy as np
import torch

import matplotlib.pyplot as plt

sys.path.append("../../")
sys.path.append("../input/modules")
from utils import down_sampler  # noqa: E402


class FeatTrainCollater(object):
    """Customized collater for Pytorch DataLoader for feat form data in training."""

    def __init__(
        self,
        max_frames=512,
        l_target=16,
        mode="sum",
        random=False,
        use_dializer=False,
        split=8,
    ):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            max_frames (int): The max size of melspectrograms frame.
            l_target (int): Length of embedding time frame.
            mode (str): Mode of down sampler. ["sum" or "binary"]
            random (bool): Use simple random frame which may contain only noise(as same sa inference condition).
            use_dializer (bool): Use frame mask for dialization.
            split (int): Ratio of contain ground truth sound.
        """
        self.max_frames = max_frames
        self.mode = mode
        self.l_target = l_target
        self.random = random
        self.use_dializer = use_dializer
        self.split = split

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of dict of melspectrogram and features.

        Returns:
            Tensor: logmel batch (B, bin, T).
            Tensor: frame label (B, T, n_class).
            Tensor: clip label (B, n_class).
        """
        logmels = [b["feats"] for b in batch]
        matrix_tp_list = [b["matrix_tp"] for b in batch]
        all_time_list = [b["time_list"] for b in batch]
        logmel_batch = []
        frame_batch = []
        clip_batch = []
        if self.use_dializer:
            frame_mask_batch = []
        # select start point
        cnt = 0
        for logmel, matrix_tp, time_list in zip(logmels, matrix_tp_list, all_time_list):
            l_spec = len(logmel)
            if self.random:
                beginning = random.randrange(0, l_spec - self.max_frames)
            else:
                idx = random.randrange(len(time_list))
                # logging.debug(f"{l_spec}, {time_list}")
                time_start = int(l_spec * time_list[idx][0] / 60)
                time_end = int(l_spec * time_list[idx][1] / 60)
                center = np.round((time_start + time_end) / 2)
                quarter = ((time_end - time_start) // self.split) * (
                    self.split // 2 - 1
                )
                beginning = center - self.max_frames - quarter
                if beginning < 0:
                    beginning = 0
                beginning = random.randrange(beginning, center + quarter)
            ending = beginning + self.max_frames
            if ending > l_spec:
                ending = l_spec
            beginning = ending - self.max_frames
            logmel_batch.append(logmel[beginning:ending].astype(np.float32))
            embedded_frame = down_sampler(
                matrix_tp[beginning:ending], l_target=self.l_target, mode=self.mode
            )
            frame_batch.append(embedded_frame.astype(np.float32))
            clip_batch.append(
                matrix_tp[beginning:ending].any(axis=0).astype(np.float32)
            )
            if self.random:
                clip_batch[-1][24] = (~clip_batch[-1][:24].any()).astype(np.float32)
            if self.use_dializer:
                frame_mask_batch.append(
                    embedded_frame[:, :24].any(axis=1).astype(np.float32)
                )
            logging.debug(
                f"sum:{clip_batch[-1].sum()}:{time_start},{time_end}: {l_spec}: {beginning},{ending}"
            )
            logging.debug(f"{clip_batch[-1]}")
            if matrix_tp.any(axis=0).sum() != 1:
                idx = np.where(clip_batch[-1])
                plt.figure(figsize=(12, 6))
                plt.subplot(2, 1, 1)
                plt.imshow(logmel.T, aspect="auto")
                plt.axvline(x=beginning, c="r")
                plt.axvline(x=ending, c="r")
                plt.axvline(x=time_start, c="y")
                plt.axvline(x=time_end, c="y")
                plt.colorbar()
                plt.subplot(2, 1, 2)
                plt.imshow(matrix_tp.T, aspect="auto")
                plt.axvline(x=beginning, c="r")
                plt.axvline(x=ending, c="r")
                plt.axvline(x=time_start, c="y")
                plt.axvline(x=time_end, c="y")
                plt.colorbar()
                plt.title(
                    f"l:{l_spec}:{beginning},{ending},{idx}{time_start},{time_end}"
                )
                plt.tight_layout()
                plt.savefig(f"tmp/cnt{cnt}.png")
                plt.close()
            cnt += 1
        # convert each batch to tensor, assume that each item in batch has the same length
        batch = {}
        # (B, mel, max_frames)
        batch["X"] = torch.tensor(logmel_batch, dtype=torch.float).transpose(2, 1)
        # (B, l_target, n_class)
        batch["y_frame"] = torch.tensor(frame_batch, dtype=torch.float)
        # (B, n_class)
        batch["y_clip"] = torch.tensor(clip_batch, dtype=torch.float)
        if self.use_dializer:
            # (B, l_target, 1)
            batch["frame_mask"] = torch.tensor(
                frame_mask_batch, dtype=torch.float
            ).unsqueeze(2)
        return batch


class FeatEvalCollater(object):
    """Customized collater for Pytorch DataLoader for feat form data in evaluation."""

    def __init__(self, max_frames=512, n_split=20, is_label=False):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            max_frames (int): The max size of melspectrograms frame.
            n_split (int): The number of split eval data to apply the model.

        """
        self.max_frames = max_frames
        self.n_split = n_split
        self.is_label = is_label

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of dict of melspectrogram and features.

        Returns:
            Tensor: Feat batch (B, bin, max_frames).
            Tensor: clip label (B, n_class).
        """
        logmels = [b["feats"] for b in batch]
        frame_lengths = np.array([logmel.shape[0] for logmel in logmels])
        hop_size = np.array(
            [
                max((frame_length - self.max_frames) // (self.n_split - 1), 1)
                for frame_length in frame_lengths
            ]
        )
        start_frames = np.array(
            [(hop_size * i).astype(np.int64) for i in range(self.n_split - 1)]
            + [frame_lengths - self.max_frames]
        )
        end_frames = start_frames + self.max_frames
        items = {}
        for i, (start_frame, end_frame) in enumerate(zip(start_frames, end_frames)):
            logmel_batch = [
                logmel[start_frame[j] : end_frame[j]]
                for j, logmel in enumerate(logmels)
            ]
            items[f"X{i}"] = torch.tensor(logmel_batch, dtype=torch.float).transpose(
                2, 1
            )  # (B, mel, max_frames)

        if self.is_label:
            matrix_tp_list = [b["matrix_tp"] for b in batch]
            items["y_clip"] = torch.tensor(
                [
                    matrix_tp.any(axis=0).astype(np.float32)
                    for matrix_tp in matrix_tp_list
                ],
                dtype=torch.float,
            )
        return items
