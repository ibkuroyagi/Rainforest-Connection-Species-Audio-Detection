#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Ibuki Kuroyanagi

"""Train Anomary Sound Detection model."""

import argparse
import logging
import os
import sys

import matplotlib
import pandas as pd
import torch
import torch.nn as nn
import yaml

from torch.utils.data import DataLoader

sys.path.append("../../")
sys.path.append("../input/modules")
from losses import CenterLoss
from utils import lwlrap
import parallel_wavegan
import parallel_wavegan.models
import parallel_wavegan.optimizers
import parallel_wavegan.losses
from datasets import RainForestDataset


# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train outliter exposure model (See detail in parallel_wavegan/bin/train.py)."
    )
    parser.add_argument(
        "--datadir",
        default=None,
        type=str,
        help="root directory.",
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="directory to save checkpoints."
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="",
        help="Paht of pretrained model's weight.",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument("--seed", type=int, default=1, help="seed.")
    parser.add_argument(
        "--n_anomaly", type=int, default=1, help="The number of max anomaly file."
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to resume training. (default="")',
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.getLogger("matplotlib.font_manager").disabled = True
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")
    if config["model_params"].get("require_prep", True):
        train_keys = ["wave"]
        eval_keys = ["wave"]
    else:
        train_keys = ["feats"]
        eval_keys = ["feats"]
    train_tp = pd.read_csv(os.path.join(args.datadir, "train_tp.csv"))
    if config.get("train_dataset_mode", "tp") == "all":
        train_fp = pd.read_csv(os.path.join(args.datadir, "train_fp.csv"))
    else:
        train_fp = None
    # get dataset
    train_dataset = RainForestDataset(
        root_dir=os.path.join(args.datadir, "train"),
        train_tp=train_tp,
        train_fp=train_fp,
        keys=train_keys,
        mode=config.get("train_dataset_mode", "tp"),
        is_normalize=config.get("is_normalize", False),
        allow_cache=config.get("allow_cache", False),  # keep compatibility
        seed=None,
    )
    logging.info(f"The number of training files = {len(train_dataset)}.")
    dev_dataset = MelASDDataset(
        pos_root_dir=args.eval_dumpdir,
        neg_root_dirs=[],
        keys=train_keys,
        machine_id=args.pos_machine_id,
        mode="train",
        seed=args.seed,
        max_anomaly=64,
        is_normalize=config.get("is_normalize", False),
        allow_cache=config.get("allow_cache", False),  # keep compatibility
    )
    logging.info(f"The number of development files = {len(dev_dataset)}.")
    eval_dataset = MelASDDataset(
        pos_root_dir=args.eval_dumpdir,
        neg_root_dirs=[],
        keys=eval_keys,
        machine_id=args.pos_machine_id,
        mode="test",
        seed=args.seed,
        max_anomaly=64,
        is_normalize=config.get("is_normalize", False),
        allow_cache=config.get("allow_cache", False),  # keep compatibility
    )
    logging.info(f"The number of evaluation files = {len(eval_dataset)}.")
    if (
        config.get("batch_sampler_type", "OECBalancedBatchSampler")
        == "OECBalancedBatchSampler"
    ):
        from parallel_wavegan.datasets import OECBalancedBatchSampler

        train_balanced_batch_sampler = OECBalancedBatchSampler(
            train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
        )
    elif (
        config.get("batch_sampler_type", "OECBalancedBatchSampler")
        == "OECBalancedWithAnomalyBatchSampler"
    ):
        from parallel_wavegan.datasets import OECBalancedWithAnomalyBatchSampler

        train_balanced_batch_sampler = OECBalancedWithAnomalyBatchSampler(
            train_dataset,
            batch_size=config["batch_size"],
            n_anomaly=min(config["n_anomaly_in_batch"], args.n_anomaly),
            shuffle=True,
            drop_last=True,
        )

    # get data loader
    if config["model_params"].get("require_prep", True):
        from parallel_wavegan.trainer import WaveEvalCollater

        eval_collater = WaveEvalCollater(
            sf=config["sampling_rate"],
            sec=config.get("sec", 4),
            n_split=config.get("n_eval_split", 3),
        )
        if args.pos_machine_id is None:
            from parallel_wavegan.trainer import WaveTrainCollater

            train_collater = WaveTrainCollater(
                sf=config["sampling_rate"],
                sec=config.get("sec", 4),
                pos_machine=args.pos_machine,
            )
        else:
            from parallel_wavegan.trainer import WaveMachineIdTrainCollater

            train_collater = WaveMachineIdTrainCollater(
                sf=config["sampling_rate"],
                sec=config.get("sec", 4),
                pos_machine=args.pos_machine,
                pos_machine_id=args.pos_machine_id,
                use_anomaly=args.n_anomaly >= 1,
            )

    else:
        from parallel_wavegan.trainer import FeatEvalCollater

        eval_collater = FeatEvalCollater(
            max_frames=config.get("max_frames", 256),
            n_split=config.get("n_eval_split", 3),
        )
        if args.pos_machine_id is None:
            from parallel_wavegan.trainer import FeatTrainCollater

            train_collater = FeatTrainCollater(
                max_frames=config.get("max_frames", 256), pos_machine=args.pos_machine
            )
        else:
            from parallel_wavegan.trainer import FeatMachineIdTrainCollater

            train_collater = FeatMachineIdTrainCollater(
                max_frames=config.get("max_frames", 256),
                pos_machine=args.pos_machine,
                pos_machine_id=args.pos_machine_id,
                use_anomaly=args.n_anomaly >= 1,
                is_label=config.get("is_label", False),
            )

    data_loader = {
        "train": DataLoader(
            dataset=train_dataset,
            batch_sampler=train_balanced_batch_sampler,
            collate_fn=train_collater,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        ),
        "dev": DataLoader(
            dataset=dev_dataset,
            # batch_sampler=dev_balanced_batch_sampler,
            collate_fn=train_collater,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        ),
        "eval": DataLoader(
            eval_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=eval_collater,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        ),
    }
    # from IPython import embed

    # embed()
    # define models and optimizers
    model_class = getattr(
        parallel_wavegan.models,
        # keep compatibility
        config.get("model_type", "ResNet38"),
    )
    model = model_class(training=True, **config["model_params"]).to(device)
    if len(args.cache_path) != 0:
        weights = torch.load(args.cache_path)
        model.load_state_dict(weights["model"])
        logging.info(f"Successfully load weight from {args.cache_path}")
    if config.get("model_type", "ResNet38") in ["ResNet38Double", "ResNet38Att"]:
        conv_params = []
        fc_param = []
        for name, param in model.named_parameters():
            if "conv_layer" in name:
                conv_params.append(param)
            else:
                fc_param.append(param)
    else:
        model.bn0 = nn.BatchNorm2d(config["num_mels"])
        model.fc_audioset = nn.Linear(**config["fc_audioset"])
        nn.init.xavier_uniform_(model.fc_audioset.weight)
        conv_params = []
        fc_param = []
        for name, param in model.named_parameters():
            if name in ["fc_audioset.weight", "fc_audioset.bias"]:
                fc_param.append(param)
            else:
                conv_params.append(param)
    loss_class = getattr(
        parallel_wavegan.losses,
        # keep compatibility
        config.get("loss_type", "BCELoss"),
    )
    criterion = loss_class(**config["loss_params"]).to(device)
    optimizer_class = getattr(
        parallel_wavegan.optimizers,
        # keep compatibility
        config.get("optimizer_type", "Adam"),
    )
    optimizer = optimizer_class(
        [
            {"params": conv_params, "lr": config["optimizer_params"]["conv_lr"]},
            {"params": fc_param, "lr": config["optimizer_params"]["fc_lr"]},
        ]
    )

    scheduler_class = getattr(
        torch.optim.lr_scheduler,
        # keep compatibility
        config.get("scheduler_type", "StepLR"),
    )
    scheduler = scheduler_class(optimizer=optimizer, **config["scheduler_params"])

    logging.info(model)
    # define trainer
    if config.get("model_type", "ResNet38") == "ResNet38":
        from parallel_wavegan.trainer import OECTrainer

        trainer = OECTrainer(
            steps=0,
            epochs=0,
            data_loader=data_loader,
            model=model.to(device),
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            train=True,
        )
    else:
        from parallel_wavegan.trainer import GreedyOECTrainer

        trainer = GreedyOECTrainer(
            steps=0,
            epochs=0,
            data_loader=data_loader,
            model=model.to(device),
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            train=True,
        )
    # resume from checkpoint
    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        logging.info(f"Successfully resumed from {args.resume}.")
    # run training loop
    try:
        trainer.run()
    except KeyboardInterrupt:
        trainer.save_checkpoint(
            os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl")
        )
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
