import logging
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch


class FeatTrainCollater(object):
    """Customized collater for Pytorch DataLoader for feat form data in training."""

    def __init__(self, max_frames=256, pos_machine="fan"):
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
        matrix_tp = [b["matrix_tp"] for b in batch]

        # make batch with random cut
        frame_lengths = [logmel.shape[0] for logmel in logmels]
        start_frames = np.array(
            [np.random.randint(fl - self.max_frames) for fl in frame_lengths]
        )
        end_frames = start_frames + self.max_frames
        logmel_batch = [
            logmel[start:end]
            for logmel, start, end in zip(logmels, start_frames, end_frames)
        ]
        machine_batch = [
            np.int64(str(machine, "utf-8") == self.pos_machine) for machine in machines
        ]

        # convert each batch to tensor, assume that each item in batch has the same length
        logmel_batch = torch.tensor(logmel_batch, dtype=torch.float).transpose(
            2, 1
        )  # (B, mel, max_frames)
        machine_batch = torch.tensor(machine_batch, dtype=torch.float).unsqueeze(
            1
        )  # (B, 1)
        return {"X": logmel_batch, "y": machine_batch}
