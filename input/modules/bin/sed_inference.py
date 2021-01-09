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
import models  # noqa: E402
from datasets import RainForestDataset  # noqa: E402
from trainers import SEDTrainer  # noqa: E402
from utils import write_hdf5  # noqa: E402
from utils import lwlrap  # noqa: E402

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
        "--dumpdirs",
        default=[],
        type=str,
        nargs="+",
        help="root dump directory.",
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="directory to save checkpoints."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument("--seed", type=int, default=1, help="seed.")
    parser.add_argument(
        "--checkpoints",
        default=[],
        type=str,
        nargs="+",
        help="List of checkpoint file path to resume training. (default=[])",
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
    config["n_TTA"] = len(args.dumpdirs)
    config.update(vars(args))
    config["n_target"] = 24
    config["trained_model_fold"] = []
    for i, checkpoint in enumerate(args.checkpoints):
        if checkpoint != "no_model":
            config["trained_model_fold"].append(i)
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")
    eval_keys = ["feats", "matrix_tp"]
    train_tp = pd.read_csv(os.path.join(args.datadir, "train_tp.csv"))
    # get dataset
    tp_list = train_tp["recording_id"].unique()
    columns = ["recording_id"] + [f"s{i}" for i in range(config["n_target"])]
    ground_truth = pd.DataFrame(
        np.zeros((len(tp_list), config["n_target"] + 1)), columns=columns
    )
    ground_truth["recording_id"] = tp_list
    for i, recording_id in enumerate(train_tp["recording_id"].values):
        ground_truth.iloc[
            ground_truth["recording_id"] == recording_id,
            train_tp.loc[i, "species_id"] + 1,
        ] = 1.0
    ground_truth_path = os.path.join(args.datadir, "ground_truth.csv")
    if not os.path.isfile(ground_truth_path):
        ground_truth.to_csv(ground_truth_path, index=False)
    kfold = MultilabelStratifiedKFold(
        n_splits=config["n_fold"], shuffle=True, random_state=config["seed"]
    )
    y = ground_truth.iloc[:, 1:].values
    # Initialize out of fold
    oof_clip = np.zeros((len(ground_truth), config["n_eval_split"], config["n_target"]))
    oof_frame = np.zeros(
        (
            len(ground_truth),
            config["n_eval_split"],
            config["l_target"],
            config["n_class"],
        )
    )
    scores = []
    # Initialize each fold prediction.
    sub = pd.read_csv(os.path.join(args.datadir, "sample_submission.csv"))
    pred_clip = np.zeros(
        (
            len(config["trained_model_fold"]),
            len(sub),
            config["n_eval_split"],
            config["n_target"],
        )
    )
    pred_frame = np.zeros(
        (
            len(config["trained_model_fold"]),
            len(sub),
            config["n_eval_split"],
            config["l_target"],
            config["n_class"],
        )
    )
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(y, y)):
        logging.info(f"Start training fold {fold}.")
        # define models and optimizers
        if fold not in config["trained_model_fold"]:
            logging.info(f"Skip fold {fold}. Due to not found trained model.")
            continue
        model_class = getattr(
            models,
            # keep compatibility
            config.get("model_type", "Cnn14_DecisionLevelAtt"),
        )
        model = model_class(training=False, **config["model_params"]).to(device)
        if config["model_type"] in ["ResNext50", "Cnn14_DecisionLevelAtt"]:
            from models import AttBlock

            model.bn0 = nn.BatchNorm2d(config["num_mels"])
            model.att_block = AttBlock(**config["att_block"])
            logging.info("Successfully initialize custom weight.")

        if fold == 0:
            logging.info(model)
        # train_y = ground_truth.iloc[train_idx]
        valid_y = ground_truth.iloc[valid_idx]
        train_tp["use_train"] = train_tp["recording_id"].map(
            lambda x: x not in valid_y["recording_id"].values
        )
        # get data loader
        if config["model_params"].get("require_prep", False):
            # from datasets import WaveEvalCollater

            # dev_collater = WaveEvalCollater(
            #     sf=config["sampling_rate"],
            #     sec=config.get("sec", 10),
            #     n_split=config.get("n_eval_split", 3),
            # )
            pass
        else:
            from datasets import FeatEvalCollater

            dev_collater = FeatEvalCollater(
                max_frames=config.get("max_frames", 512),
                n_split=config.get("n_eval_split", 20),
                is_label=True,
            )
        tta_oof_clip = np.zeros(
            (
                config["n_TTA"],
                len(valid_idx),
                config["n_eval_split"],
                config["n_target"],
            )
        )
        tta_oof_frame = np.zeros(
            (
                config["n_TTA"],
                len(valid_idx),
                config["n_eval_split"],
                config["l_target"],
                config["n_class"],
            )
        )
        tta_scores = np.zeros(config["n_TTA"])
        # Initialize each fold prediction.
        tta_pred_clip = np.zeros(
            (
                config["n_TTA"],
                len(sub),
                config["n_eval_split"],
                config["n_target"],
            )
        )
        tta_pred_frame = np.zeros(
            (
                config["n_TTA"],
                len(sub),
                config["n_eval_split"],
                config["l_target"],
                config["n_class"],
            )
        )
        for i, dumpdir in enumerate(args.dumpdirs):
            valid_dataset = RainForestDataset(
                files=[
                    os.path.join(dumpdir, "train", f"{recording_id}.h5")
                    for recording_id in tp_list[valid_idx]
                ],
                keys=eval_keys,
                mode="test",
                is_normalize=config.get("is_normalize", False),
                allow_cache=False,
                seed=None,
            )
            logging.info(f"The number of validation files = {len(valid_dataset)}.")

            data_loader = {
                "eval": DataLoader(
                    valid_dataset,
                    batch_size=config["batch_size"],
                    shuffle=False,
                    collate_fn=dev_collater,
                    num_workers=config["num_workers"],
                    pin_memory=config["pin_memory"],
                ),
            }
            # define valid trainer
            trainer = SEDTrainer(
                steps=0,
                epochs=0,
                data_loader=data_loader,
                model=model.to(device),
                criterion={},
                optimizer={},
                scheduler={},
                config=config,
                device=device,
                train=False,
                use_center_loss=config.get("use_center_loss", False),
                use_dializer=config.get("use_dializer", False),
                save_name=f"fold{fold}",
            )
            trainer.load_checkpoint(args.checkpoints[fold])
            logging.info(
                f"Successfully resumed from {args.checkpoints[fold]}.(Epochs:{trainer.epochs}, Steps:{trainer.steps})"
            )
            # inference validation data
            oof_dict = trainer.inference(mode="valid")
            tta_oof_clip[i] = oof_dict["y_clip"][:, :, : config["n_target"]]
            tta_oof_frame[i] = oof_dict["y_frame"]
            tta_scores[i] = oof_dict["score"]
            logging.info(f"Fold:{fold},TTA:{i} lwlrap:{tta_scores[i]:.6f}")
            # initialize test data
            test_dataset = RainForestDataset(
                root_dirs=[os.path.join(dumpdir, "test")],
                keys=["feats"],
                mode="test",
                is_normalize=config.get("is_normalize", False),
                allow_cache=False,
                seed=None,
            )
            logging.info(f"The number of test files = {len(test_dataset)}.")
            if config["model_params"].get("require_prep", False):
                # from datasets import WaveEvalCollater

                # eval_collater = WaveEvalCollater(
                #     sf=config["sampling_rate"],
                #     sec=config.get("sec", 10),
                #     n_split=config.get("n_eval_split", 3),
                # )
                pass
            else:
                from datasets import FeatEvalCollater

                eval_collater = FeatEvalCollater(
                    max_frames=config.get("max_frames", 512),
                    n_split=config.get("n_eval_split", 20),
                    is_label=False,
                )
            data_loader = {
                "eval": DataLoader(
                    test_dataset,
                    batch_size=config["batch_size"],
                    shuffle=False,
                    collate_fn=eval_collater,
                    num_workers=config["num_workers"],
                    pin_memory=config["pin_memory"],
                ),
            }
            # define valid trainer
            trainer = SEDTrainer(
                steps=0,
                epochs=0,
                data_loader=data_loader,
                model=model.to(device),
                criterion={},
                optimizer={},
                scheduler={},
                config=config,
                device=device,
                train=False,
                use_center_loss=config.get("use_center_loss", False),
                use_dializer=config.get("use_dializer", False),
                save_name=f"fold{fold}",
            )
            trainer.load_checkpoint(args.checkpoints[fold])
            logging.info(
                f"Successfully resumed from {args.checkpoints[fold]}.(Epochs:{trainer.epochs}, Steps:{trainer.steps})"
            )
            # inference test data
            pred_dict = trainer.inference(mode="test")
            tta_pred_clip[i] = pred_dict["y_clip"][:, :, : config["n_target"]]
            tta_pred_frame[i] = pred_dict["y_frame"]

            logging.info(f"Fold:{fold},TTA:{i} Successfully inference test data.")
        oof_clip[valid_idx] = tta_oof_clip.mean(axis=0)
        oof_frame[valid_idx] = tta_oof_frame.mean(axis=0)
        scores.append(tta_scores.mean())
        logging.info(f"Fold:{fold}, lwlrap:{scores[-1]:.6f}")
        pred_clip[fold] = tta_pred_clip.mean(axis=0)
        pred_frame[fold] = tta_pred_frame.mean(axis=0)

    # save inference results
    write_hdf5(
        os.path.join(args.outdir, "oof.h5"),
        "y_clip",
        oof_clip.astype(np.float32),
    )
    write_hdf5(
        os.path.join(args.outdir, "oof.h5"),
        "y_frame",
        oof_frame.astype(np.float32),
    )
    logging.info(f"Successfully saved oof at {os.path.join(args.outdir, 'oof.h5')}.")
    pred_clip_mean = pred_clip.mean(axis=0)
    pred_frame_mean = pred_frame.mean(axis=0)
    write_hdf5(
        os.path.join(args.outdir, "pred.h5"),
        "y_clip",
        pred_clip_mean.astype(np.float32),
    )
    write_hdf5(
        os.path.join(args.outdir, "pred.h5"),
        "y_frame",
        pred_frame_mean.astype(np.float32),
    )
    logging.info(f"Successfully saved pred at {os.path.join(args.outdir, 'pred.h5')}.")

    # modify submission shape
    oof_sub = ground_truth.copy()
    oof_sub.iloc[:, 1:] = oof_clip.max(axis=1)
    oof_sub.to_csv(os.path.join(args.outdir, "oof.csv"), index=False)
    logging.info(f"Successfully saved oof at {os.path.join(args.outdir, 'oof.csv')}.")
    oof_score = lwlrap(ground_truth.iloc[:, 1:].values, oof_sub.iloc[:, 1:].values)
    for i, score in enumerate(scores):
        logging.info(f"Fold:{i} oof score is {score:.6f}")
    logging.info(f"Average oof score is {np.array(scores).mean():.6f}")
    logging.info(f"All oof score is {oof_score:.6f}")

    sub.iloc[:, 1:] = pred_clip_mean.max(axis=1)
    sub.to_csv(os.path.join(args.outdir, "submission.csv"), index=False)
    logging.info(
        f"Successfully saved submission at {os.path.join(args.outdir, 'submission.csv')}."
    )


if __name__ == "__main__":
    main()
