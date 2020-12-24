#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Ibuki Kuroyanagi

"""Train Sound Event Detection model."""

import argparse
import logging
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from torch.utils.data import DataLoader

sys.path.append("../../")
sys.path.append("../input/modules")
import losses  # noqa: E402
import models  # noqa: E402
import optimizers  # noqa: E402
from datasets import RainForestDataset  # noqa: E402

sys.path.append("../input/iterative-stratification-master")
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold  # noqa: E402

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
        help="root data directory.",
    )
    parser.add_argument(
        "--dumpdir",
        default=None,
        type=str,
        help="root dump directory.",
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
    if (config["model_params"].get("require_prep", False)) or (
        config.get("wave_mode", False)
    ):
        train_keys = ["wave"]
        eval_keys = ["wave", "matrix_tp"]
    else:
        train_keys = ["feats"]
        eval_keys = ["feats", "matrix_tp"]
    train_tp = pd.read_csv(os.path.join(args.datadir, "train_tp.csv"))
    if config.get("train_dataset_mode", "tp") == "all":
        train_fp = pd.read_csv(os.path.join(args.datadir, "train_fp.csv"))
    else:
        train_fp = None
    # get dataset
    tp_list = train_tp["recording_id"].unique()
    columns = ["recording_id"] + [f"s{i}" for i in range(24)]
    ground_truth = pd.DataFrame(np.zeros((len(tp_list), 25)), columns=columns)
    ground_truth["recording_id"] = tp_list
    for i, recording_id in enumerate(train_tp["recording_id"].values):
        ground_truth.iloc[
            ground_truth["recording_id"] == recording_id,
            train_tp.loc[i, "species_id"] + 1,
        ] = 1.0
    kfold = MultilabelStratifiedKFold(
        n_splits=config["n_fold"], shuffle=True, random_state=config["seed"]
    )
    y = ground_truth.iloc[:, 1:].values
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(y, y)):
        logging.info(f"Start training fold {fold}.")
        # train_y = ground_truth.iloc[train_idx]
        valid_y = ground_truth.iloc[valid_idx]
        train_tp["use_train"] = train_tp["recording_id"].map(
            lambda x: x not in valid_y["recording_id"].values
        )
        train_dataset = RainForestDataset(
            root_dir=os.path.join(args.dumpdir, "train"),
            train_tp=train_tp[train_tp["use_train"]],
            train_fp=train_fp,
            keys=train_keys,
            mode=config.get("train_dataset_mode", "tp"),
            is_normalize=config.get("is_normalize", False),
            allow_cache=config.get("allow_cache", False),  # keep compatibility
            seed=None,
        )
        logging.info(f"The number of training files = {len(train_dataset)}.")
        dev_dataset = RainForestDataset(
            root_dir=os.path.join(args.dumpdir, "train"),
            train_tp=train_tp[~train_tp["use_train"]],
            train_fp=train_fp,
            keys=train_keys,
            mode=config.get("train_dataset_mode", "tp"),
            is_normalize=config.get("is_normalize", False),
            allow_cache=config.get("allow_cache", False),  # keep compatibility
            seed=None,
        )
        logging.info(f"The number of development files = {len(dev_dataset)}.")
        eval_dataset = RainForestDataset(
            files=[
                os.path.join(args.dumpdir, "train", f"{recording_id}.h5")
                for recording_id in tp_list[valid_idx]
            ],
            keys=eval_keys,
            mode="test",
            is_normalize=config.get("is_normalize", False),
            allow_cache=config.get("allow_cache", False),  # keep compatibility
            seed=None,
        )
        logging.info(f"The number of evaluation files = {len(eval_dataset)}.")

        # get data loader
        if config["model_params"].get("require_prep", False):
            # from datasets import WaveEvalCollater

            # eval_collater = WaveEvalCollater(
            #     sf=config["sampling_rate"],
            #     sec=config.get("sec", 10),
            #     n_split=config.get("n_eval_split", 3),
            # )
            # from datasets import WaveTrainCollater

            # train_collater = WaveTrainCollater(
            #     sf=config["sampling_rate"],
            #     sec=config.get("sec", 10),
            # )
            pass
        else:
            from datasets import FeatEvalCollater

            eval_collater = FeatEvalCollater(
                max_frames=config.get("max_frames", 512),
                n_split=config.get("n_eval_split", 20),
                is_label=True,
            )
            from datasets import FeatTrainCollater

            train_collater = FeatTrainCollater(
                max_frames=config.get("max_frames", 512),
                l_target=config.get("l_target", 16),
                mode=config.get("collater_mode", "sum"),
            )
        data_loader = {
            "train": DataLoader(
                dataset=train_dataset,
                collate_fn=train_collater,
                batch_size=config["batch_size"],
                shuffle=True,
                drop_last=True,
                num_workers=config["num_workers"],
                pin_memory=config["pin_memory"],
            ),
            "dev": DataLoader(
                dataset=dev_dataset,
                collate_fn=train_collater,
                shuffle=False,
                batch_size=config["batch_size"],
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
            models,
            # keep compatibility
            config.get("model_type", "Cnn14_DecisionLevelAtt"),
        )
        model = model_class(training=True, **config["model_params"]).to(device)
        if len(args.cache_path) != 0:
            weights = torch.load(args.cache_path)
            model.load_state_dict(weights["model"])
            logging.info(f"Successfully load weight from {args.cache_path}")
        if config.get("model_type" "Cnn14_DecisionLevelAtt") in [
            "ResNet38Double",
            "ResNet38Att",
        ]:
            conv_params = []
            fc_param = []
            for name, param in model.named_parameters():
                if "conv_layer" in name:
                    conv_params.append(param)
                else:
                    fc_param.append(param)
        else:
            from models import AttBlock

            model.bn0 = nn.BatchNorm2d(config["num_mels"])
            model.att_block = AttBlock(**config["att_block"])
            nn.init.xavier_uniform_(model.att_block.att.weight)
            nn.init.xavier_uniform_(model.att_block.cla.weight)
            logging.info("Successfully initialize custom weight.")
            conv_params = []
            fc_param = []
            for name, param in model.named_parameters():
                if name.startswith(("resnext50.fc", "fc1", "att_block")):
                    fc_param.append(param)
                else:
                    conv_params.append(param)
        loss_class = getattr(
            losses,
            # keep compatibility
            config.get("loss_type", "BCEWithLogitsLoss"),
        )
        criterion = loss_class(**config["loss_params"]).to(device)
        optimizer_class = getattr(
            optimizers,
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
        if fold == 0:
            logging.info(model)
        # define trainer
        from trainers import SEDTrainer

        trainer = SEDTrainer(
            steps=0,
            epochs=0,
            data_loader=data_loader,
            model=model.to(device),
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            train=fold == 0,
            use_center_loss=config.get("use_center_loss", False),
            save_name=f"fold{fold}",
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
                os.path.join(
                    config["outdir"], f"checkpoint-{trainer.steps}stepsfold{fold}.pkl"
                )
            )
            logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
