# -*- coding: utf-8 -*-

# Copyright 2020 Ibuki Kuroyanagi
#  MIT License (https://opensource.org/licenses/MIT)

"""Collater function modules."""

import numpy as np
import torch


class FeatTrainCollater(object):
    """Customized collater for Pytorch DataLoader for feat form data in training."""

    def __init__(self, max_frames=512):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            max_frames (int): The max size of melspectrograms frame.
        """
        self.max_frames = max_frames

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
