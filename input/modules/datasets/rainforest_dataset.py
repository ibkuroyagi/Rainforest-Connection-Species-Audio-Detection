# -*- coding: utf-8 -*-

# Copyright 2020 Ibuki Kuroyanagi

import logging
import sys
import h5py
import numpy as np
from multiprocessing import Manager

# from torch.utils.data.sampler import BatchSampler
from torch.utils.data import Dataset

sys.path.append("../../")
sys.path.append("../input/modules")
import datasets  # noqa: E402
from utils import find_files  # noqa: E402
from utils import logmelfilterbank  # noqa: E402


class RainForestDataset(Dataset):
    """PyTorch compatible mel dataset."""

    def __init__(
        self,
        root_dirs=[],
        files=None,
        train_tp=None,
        train_fp=None,
        keys=["feats"],
        mode="tp",
        is_normalize=False,
        allow_cache=False,
        config={},
        seed=None,
    ):
        """Initialize dataset.

        Args:
            root_dirs (list): List of root directories for dumped files.
            train_tp (DataFrame): train_tp (default: None)
            train_fp (DataFrame): train_fp (default: None)
            keys: (list): List of key of dataset.
            mode (list): Mode of dataset. [tp, all, test]
            is_normalize(bool): flag of normalize
            allow_cache (bool): Whether to allow cache of the loaded files.
            config (dict): Setting dict for on the fly.
            seed (int): seed
        """
        # if seed is not None:
        #     self.seed = seed
        #     np.random.seed(seed)
        # find all of the mel files
        if (files is None) and (len(root_dirs) != 0):
            files = []
            for root_dir in root_dirs:
                files += sorted(find_files(root_dir, "*.h5"))
        use_file_keys = []
        use_file_list = []
        use_time_list = []
        if mode == "tp":
            tp_list = train_tp["recording_id"].unique()
            for file in files:
                recording_id = file.split("/")[-1].split(".")[0]
                if recording_id in tp_list:
                    use_file_keys.append(keys + ["matrix_tp"])
                    use_file_list.append(file)
                    # logging.debug(f"{facter}: {file}")
                    use_time_list.append(
                        train_tp[train_tp["recording_id"] == recording_id]
                        .loc[:, ["t_min", "t_max"]]
                        .values
                    )
        elif mode == "all":
            tp_list = train_tp["recording_id"].unique()
            fp_list = train_fp["recording_id"].unique()
            for file in files:
                recording_id = file.split("/")[-1].split(".")[0]
                if recording_id in tp_list:
                    use_file_keys.append(keys + ["matrix_tp"])
                    use_file_list.append(file)
                    # logging.debug(f"{facter}: {file}")
                    use_time_list.append(
                        train_tp[train_tp["recording_id"] == recording_id]
                        .loc[:, ["t_min", "t_max"]]
                        .values
                    )
                if recording_id in fp_list:
                    use_file_keys.append(keys + ["matrix_fp"])
                    use_file_list.append(file)
                    use_time_list.append(
                        train_fp[train_fp["recording_id"] == recording_id]
                        .loc[:, ["t_min", "t_max"]]
                        .values
                    )
        elif mode == "test":
            for file in files:
                use_file_keys.append(keys)
                use_file_list.append(file)
        self.use_file_keys = use_file_keys
        self.use_file_list = use_file_list
        self.use_time_list = use_time_list
        self.mode = mode
        self.allow_cache = allow_cache
        self.is_normalize = is_normalize
        self.config = config
        self.transform = None
        if ("wave" in use_file_keys) and (
            config.get("augmentation_params", None) is not None
        ):
            compose_list = []
            for key in self.config["augmentation_params"].keys():
                aug_class = getattr(
                    datasets,
                    key,
                )
                compose_list.append(
                    aug_class(**self.config["augmentation_params"][key])
                )
                logging.debug(f"{key}")
            self.transform = datasets.Compose(compose_list)
        if allow_cache:
            # NOTE(ibuki): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(use_file_list))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            items: Dict
                wave: (ndarray) Wave (T, ).
                feats: (ndarray) Feature (T', C).
                matrix_tp: (ndrray) Matrix of ground truth.
                time_list: (ndrray) (n_recoding_id, t_max, t_min).
        """
        if self.allow_cache and (len(self.caches[idx]) != 0):
            return self.caches[idx]
        hdf5_file = h5py.File(self.use_file_list[idx], "r")
        items = {}
        for key in self.use_file_keys[idx]:
            items[key] = hdf5_file[key][()]
        hdf5_file.close()

        if self.is_normalize and "feats" in self.use_file_keys[idx]:
            items["feats"] = (
                items["feats"] - items["feats"].mean(axis=0, keepdims=True)
            ) / items["feats"].std(axis=0, keepdims=True)
        if self.is_normalize and "wave" in self.use_file_keys:
            items["wave"] = (items["wave"] - items["wave"].mean()) / items["wave"].std()
        if self.config.get("wave_mode", False) and ("wave" in self.use_file_keys):
            items["feats"] = self.wave2spec(items["wave"])
            del items["wave"]
        if (self.mode == "all") or (self.mode == "tp"):
            items["time_list"] = self.use_time_list[idx]
        if self.allow_cache:
            self.caches[idx] = items
        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.
        """
        return len(self.use_file_list)

    def wave2spec(self, wave: np.ndarray):
        """Transfom wave to log mel sprctrogram and apply augmentation.

        Args:
            wave (ndarray): Original wave form (T,)
        Returns:
            feats (ndarray): Augmented log mel spectrogram(T', mel).
        """
        if self.transform is not None:
            wave = self.transform(wave)
        feats = logmelfilterbank(
            wave,
            sampling_rate=self.config["sr"],
            hop_size=self.config["hop_size"],
            fft_size=self.config["fft_size"],
            window=self.config["window"],
            num_mels=self.config["num_mels"],
            fmin=self.config["fmin"],
            fmax=self.config["fmax"],
        )
        return feats
