import logging
import random
import sys
import h5py
import numpy as np
from multiprocessing import Manager
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import Dataset

sys.path.append("../../")
sys.path.append("../input/modules")
from utils.utils import find_files


class RainForestDataset(Dataset):
    """PyTorch compatible mel dataset."""

    def __init__(
        self,
        root_dir,
        tp_list=[],
        fp_list=[],
        keys=["feats"],
        mode="tp",
        is_normalize=False,
        statistic={}
        allow_cache=False,
        seed=None,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory for dumped files.
            mode (list): Mode of dataset. [tp, all, test]
            is_normalize(bool): flag of normalize
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)
        # find all of the mel files
        files = sorted(find_files(root_dir, "*.h5"))
        use_file_keys = []
        use_file_list = []
        if mode == "tp":
            for file in files:
                if file.split("/")[-1].split(".")[0] in tp_list:
                    use_file_keys.append(keys + ["matrix_tp"])
                    use_file_list.append(file)
        elif mode == "all":
            for file in files:
                if file.split("/")[-1].split(".")[0] in tp_list:
                    use_file_keys.append(keys + ["matrix_tp"])
                    use_file_list.append(file)
                if file.split("/")[-1].split(".")[0] in fp_list:
                    use_file_keys.append(keys + ["matrix_fp"])
                    use_file_list.append(file)
        elif mode == "test":
            for file in files:
                use_file_keys.append(keys)
                use_file_list.append(file)
        self.use_file_keys = use_file_keys
        self.use_file_list = use_file_list
        self.mode = mode
        self.allow_cache = allow_cache
        self.is_noamalize = is_noamalize
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
                machine: (str) Name of machine.
                machine_id: (int) Number of machine id.
                is_normal: (int) Whether normal or not. (if normal == 1)
        """
        if self.allow_cache and len(self.caches[idx]) != 0:
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
        if self.is_normalize and "wave" in self.keys:
            items["wave"] = (items["wave"] - items["wave"].mean()) / items["wave"].std()

        if self.allow_cache:
            self.caches[idx] = items
        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.use_file_list)
