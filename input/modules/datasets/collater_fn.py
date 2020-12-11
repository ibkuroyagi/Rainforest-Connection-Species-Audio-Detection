# -*- coding: utf-8 -*-

# Copyright 2020 Ibuki Kuroyanagi
#  MIT License (https://opensource.org/licenses/MIT)

"""Collater function modules."""

import numpy as np
import torch


class FeatTrainCollater(object):
    """Customized collater for Pytorch DataLoader for feat form data in training."""

    def __init__(self, max_frames=512, pos_machine="fan"):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            max_frames (int): The max size of melspectrograms frame.
            pos_machine (str): The name of positive machine.

        """
        self.max_frames = max_frames
        self.pos_machine = pos_machine

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of dict of melspectrogram and features.

        Returns:
            Tensor: Feat batch (B, T, bin).
            Tensor: Machine label (B, T, n_class).

        """
        logmels = [b["feats"] for b in batch]
        matrix_tp_list = [b["matrix_tp"] for b in batch]
        all_time_list = [b["time_list"] for b in batch]
        logmel_batch = []
        frame_batch = []
        clip_batch = []
        # select start point
        for logmel, matrix_tp, time_list in zip(logmels, matrix_tp_list, all_time_list):
            l_spec = len(logmel)
            idx = np.random.randint(len(time_list))
            time_start = int(l_spec * time_list[idx][1] / 60.0)
            time_end = int(l_spec * time_list[idx][2] / 60.0)
            center = np.round((time_start + time_end) / 2)
            beginning = center - self.max_frames / 2
            if beginning < 0:
                beginning = 0
            beginning = np.random.randint(beginning, center)
            ending = beginning + self.max_frames
            if ending > l_spec:
                ending = l_spec
            beginning = ending - self.max_frames
            logmel_batch.append(logmel[beginning:ending].astype(np.float32))
            frame_batch.append(matrix_tp[beginning:ending].astype(np.float32))
            clip_batch.append(
                matrix_tp[beginning:ending].any(axis=0).astype(np.float32)
            )

        # convert each batch to tensor, assume that each item in batch has the same length
        # (B, mel, max_frames)
        logmel_batch = torch.tensor(logmel_batch, dtype=torch.float).transpose(2, 1)
        # (B, max_frame, n_class)
        frame_batch = torch.tensor(frame_batch, dtype=torch.float)
        # (B, n_class)
        clip_batch = torch.tensor(clip_batch, dtype=torch.float)
        return {"X": logmel_batch, "y_frame": frame_batch, "y_clip": clip_batch}
