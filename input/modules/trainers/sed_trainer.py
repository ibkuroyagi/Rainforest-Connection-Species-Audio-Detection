import logging
import os
import sys
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.manifold import TSNE
from tensorboardX import SummaryWriter

sys.path.append("../../")
sys.path.append("../input/modules")
from losses import FrameClipLoss
from losses import CenterLoss


class GreedyOECTrainer(object):
    """Customized trainer module for OEC training."""

    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        config,
        device=torch.device("cpu"),
        train=False,
        use_center_loss=False,
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "generator" and "discriminator" models.
            criterion (torch.nn): It must contrain "stft" and "mse" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" and "discriminator" optimizers.
            scheduler (dict): Dict of schedulers. It must contrain "generator" and "discriminator" schedulers.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.
            train (bool): Select mode of trainer.
            use_center_loss(bool): Select whether to use center loss.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.train = train
        if train:
            self.writer = SummaryWriter(config["outdir"])
        self.use_center_loss = use_center_loss
        if use_center_loss:
            self.center_loss = CenterLoss(device=device, **config["center_loss_params"])
            self.optimizer_centloss = optimizer(
                self.center_loss.parameters(), **config["optimizer_params"]
            )
            self.train_label_epoch = np.empty((0, 1))
            self.dev_label_epoch = np.empty((0, 1))
            self.train_embedding_epoch = np.empty(
                (0, config["center_loss_params"]["feat_dim"])
            )
            self.dev_embedding_epoch = np.empty(
                (0, config["center_loss_params"]["feat_dim"])
            )
            self.tsne = TSNE(**config["tsne_params"])

        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)
        self.epoch_train_loss = defaultdict(float)
        self.epoch_eval_loss = defaultdict(float)
        self.eval_metric = defaultdict(float)
        self.train_pred_epoch = np.empty((0, 1))
        self.train_y_epoch = np.empty((0, 1))
        self.dev_pred_epoch = np.empty((0, 1))
        self.dev_y_epoch = np.empty((0, 1))
        self.n_eval_split = config["n_eval_split"]
        self.forward_count = 0

    def run(self):
        """Run training."""
        self.tqdm = tqdm(
            initial=self.steps, total=self.config["train_max_steps"], desc="[train]"
        )
        while True:
            # train one epoch
            self._train_epoch()
            self._eval_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "steps": self.steps,
            "epochs": self.epochs,
        }
        state_dict["model"] = self.model.state_dict()

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state_dict["model"])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        x = batch["X"].to(self.device)
        y_frame = batch["y_frame"].to(self.device)
        y_clip = batch["y_clip"].to(self.device)
        batch_size = x.size(0)
        y_ = self.model(x)  # {y_frame: (B, T', n_class), y_clip: (B, n_class)}
        if self.config["loss_type"] == "FrameClipLoss":
            loss = self.criterion(y_["y_frame"], y_frame, y_["y_clip"], y_clip)
        if self.use_center_loss:
            center_loss_label = batch["label"]
            loss += (
                self.center_loss(y_["embedding"], center_loss_label)
                * self.config["center_loss_alpha"]
            )
            self.optimizer_centloss.zero_grad()
        if not torch.isnan(loss):
            loss = loss / self.config["accum_grads"]
            loss.backward()
            if self.use_center_loss:
                # multiple (1./alpha) in order to remove the effect of alpha on updating centers
                for param in self.center_loss.parameters():
                    param.grad.data *= 1.0 / self.config["center_loss_alpha"]
                self.optimizer_centloss.step()
            self.forward_count += 1
            if self.forward_count == self.config["accum_grads"]:
                # update parameters
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.forward_count = 0

                # update scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()

                # update counts
                self.steps += 1
                self.tqdm.update(1)
                self._check_train_finish()

        self.total_train_loss["train/loss"] += (
            loss.item() / batch_size * self.config["accum_grads"]
        )
        if self.config["loss_type"] == "FrameClipLoss":
            self.train_pred_epoch = np.concatenate(
                [self.train_pred_epoch, y_["y_clip"].detach().cpu().numpy()], axis=0
            )
            self.train_y_epoch = np.concatenate(
                [self.train_y_epoch, y_clip.detach().cpu().numpy()], axis=0
            )
        if self.use_center_loss:
            self.train_label_epoch = np.concatenate(
                [self.train_label_epoch, center_loss_label], axis=0
            )
            self.train_embedding_epoch = np.concatenate(
                [self.train_embedding_epoch, y_["embedding"].detach().cpu().numpy()]
            )

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            if self.config["rank"] == 0:
                self._check_log_interval()
                self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return
        try:
            self.epoch_train_loss["train/epoch_auc"] = roc_auc_score(
                y_true=self.train_y_epoch, y_score=self.train_pred_epoch
            )
            preds = self.train_pred_epoch > 0.5
            self.epoch_train_loss["train/epoch_acc"] = accuracy_score(
                self.train_y_epoch, preds
            )
            self.epoch_train_loss["train/epoch_recall"] = recall_score(
                self.train_y_epoch, preds
            )
            self.epoch_train_loss["train/epoch_precision"] = precision_score(
                self.train_y_epoch, preds, zero_division=0
            )
        except ValueError:
            logging.warning("Raise ValueError: May be contain NaN in y_pred.")
            pass
        # log
        logging.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({train_steps_per_epoch} steps per epoch)."
        )
        for key in self.epoch_train_loss.keys():
            logging.info(
                f"(Epoch: {self.epochs}) {key} = {self.epoch_train_loss[key]:.4f}."
            )
        self._write_to_tensorboard(self.epoch_train_loss)
        if self.use_center_loss and (self.epochs % 20 == 0):
            self.plot_embedding(
                self.train_embedding_epoch, self.train_label_epoch, name="train"
            )
        # update
        self.train_steps_per_epoch = train_steps_per_epoch
        self.epochs += 1
        # reset
        self.train_y_epoch = np.empty((0, 1))
        self.train_pred_epoch = np.empty((0, 1))
        self.train_embedding_epoch = np.empty(
            (0, self.config["center_loss_params"]["feat_dim"])
        )
        self.train_label_epoch = np.empty((0, 1))
        self.epoch_train_loss = defaultdict(float)

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        x = batch["X"].to(self.device)
        y = batch["y"].to(self.device)
        label = batch["label"]
        batch_size = x.size(0)
        y_ = self.model(x)
        if self.config["loss_type"] == "BCELoss":
            loss = self.criterion(torch.sigmoid(y_["outliter"]), y)
        elif self.config["loss_type"] == "GreedyLoss":
            loss = self.criterion(y_["outliter"], y_["anomaly"], y, label)
        else:
            loss = self.criterion(y_["outliter"], y)
        if self.use_center_loss:
            center_loss_label = torch.zeros(len(label))
            center_loss_label[label == "normal"] = 1
            loss += (
                self.center_loss(y_["embedding"], center_loss_label)
                * self.config["center_loss_alpha"]
            )
        # add to total eval loss
        self.total_eval_loss["dev/loss"] += loss.item() / batch_size
        if self.config["loss_type"] == "GreedyLoss":
            outliter_idx, anomaly_idx = get_greedy_idx(
                label=label, mode=self.config.get("greedy_mode", "all")
            )
            pred_outliter = (
                torch.sigmoid(y_["outliter"][outliter_idx]).detach().cpu().numpy()
            )
            pred_anomaly = (
                torch.sigmoid(y_["anomaly"][anomaly_idx]).detach().cpu().numpy()
            )
            self.dev_pred_epoch = np.concatenate(
                [self.dev_pred_epoch, pred_outliter, pred_anomaly], axis=0
            )
            y = y.detach().cpu().numpy()
            self.dev_y_epoch = np.concatenate(
                [self.dev_y_epoch, y[outliter_idx], y[anomaly_idx]], axis=0
            )
        else:
            y_ = torch.sigmoid(y_["outliter"]).cpu().numpy()
            self.dev_pred_epoch = np.concatenate([self.dev_pred_epoch, y_], axis=0)
            y = y.cpu().numpy()
            self.dev_y_epoch = np.concatenate([self.dev_y_epoch, y], axis=0)
        if self.use_center_loss:
            self.dev_label_epoch = np.concatenate([self.dev_label_epoch, label], axis=0)
            self.dev_embedding_epoch = np.concatenate(
                [self.dev_embedding_epoch, y_["embedding"].detach().cpu().numpy()]
            )

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start dev data's evaluation.")
        # change mode
        self.model.eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.data_loader["dev"], desc="[dev]"), 1
        ):
            # eval one step
            self._eval_step(batch)
        try:
            self.epoch_eval_loss["dev/epoch_auc"] = roc_auc_score(
                y_true=self.dev_y_epoch, y_score=self.dev_pred_epoch
            )
            preds = self.dev_pred_epoch > 0.5
            self.epoch_eval_loss["dev/epoch_acc"] = accuracy_score(
                self.dev_y_epoch, preds
            )
            self.epoch_eval_loss["dev/epoch_recall"] = recall_score(
                self.dev_y_epoch, preds
            )
            self.epoch_eval_loss["dev/epoch_precision"] = precision_score(
                self.dev_y_epoch, preds, zero_division=0
            )
        except ValueError:
            logging.warning("Raise ValueError: May be contain NaN in y_pred.")
            pass
        # log
        logging.info(
            f"(Steps: {self.steps}) Finished dev data's evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )
        for key in self.epoch_eval_loss.keys():
            logging.info(
                f"(Epoch: {self.epochs}) {key} = {self.epoch_eval_loss[key]:.4f}."
            )
        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logging.info(
                f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}."
            )
        logging.info(f"(Steps: {self.steps}) Start eval data's evaluation.")
        # record
        self._write_to_tensorboard(self.total_eval_loss)
        self._write_to_tensorboard(self.epoch_eval_loss)
        if self.use_center_loss and (self.epochs % 20 == 0):
            self.plot_embedding(
                self.dev_embedding_epoch, self.dev_label_epoch, name="dev"
            )

        # reset
        self.total_eval_loss = defaultdict(float)
        self.epoch_eval_loss = defaultdict(float)
        self.dev_pred_epoch = np.empty((0, 1))
        self.dev_y_epoch = np.empty((0, 1))
        self.dev_embedding_epoch = np.empty(
            (0, self.config["center_loss_params"]["feat_dim"])
        )
        self.dev_label_epoch = np.empty((0, 1))
        # restore mode
        self.model.train()

    def eval_score(self, save_csv="", alpha=0.5):
        """Evaluate and save intermediate result."""
        # evaluate
        keys_list = [f"X{i}" for i in range(self.n_eval_split)]
        if self.config["loss_type"] == "GreedyLoss":
            y_outliter_preds = [
                torch.empty((0, 1)).to(self.device) for _ in range(self.n_eval_split)
            ]
            y_anomaly_preds = [
                torch.empty((0, 1)).to(self.device) for _ in range(self.n_eval_split)
            ]
        else:
            y_preds = [
                torch.empty((0, 1)).to(self.device) for _ in range(self.n_eval_split)
            ]
        is_normal = torch.empty((0, 1))
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.data_loader["eval"]):
                is_normal = torch.cat([is_normal, batch["is_normal"]], dim=0)
                x_batchs = [batch[key].to(self.device) for key in keys_list]
                for i in range(self.n_eval_split):
                    y_batch_ = self.model(x_batchs[i])
                    if self.config["loss_type"] == "GreedyLoss":
                        y_outliter_preds[i] = torch.cat(
                            [y_outliter_preds[i], y_batch_["outliter"]], dim=0
                        )
                        y_anomaly_preds[i] = torch.cat(
                            [y_anomaly_preds[i], y_batch_["anomaly"]], dim=0
                        )
                    else:
                        y_preds[i] = torch.cat(
                            [y_preds[i], y_batch_["outliter"]], dim=0
                        )
        y_true = torch.ones_like(is_normal).to(self.device)

        eval_size = y_true.shape[0]
        for i in range(self.n_eval_split):
            if self.config["loss_type"] == "GreedyLoss":
                y_outliter_preds[i] = torch.sigmoid(y_outliter_preds[i])
                self.eval_metric[f"eval_metric/acc_outliter{i}"] = (
                    (y_outliter_preds[i] > 0.5).int() == y_true
                ).float().sum() / eval_size
                y_anomaly_preds[i] = torch.sigmoid(y_anomaly_preds[i])
                self.eval_metric[f"eval_metric/acc_anomaly{i}"] = (
                    (y_anomaly_preds[i] > 0.5).int() == y_true
                ).float().sum() / eval_size
            else:
                if self.config["loss_type"] == "BCELoss":
                    loss = self.criterion(torch.sigmoid(y_preds[i]), y_true)
                else:
                    loss = self.criterion(y_preds[i], y_true)
                y_preds[i] = torch.sigmoid(y_preds[i])
                self.eval_metric[f"eval_metric/loss{i}"] = loss.item() / eval_size
                self.eval_metric[f"eval_metric/acc{i}"] = (
                    (y_preds[i] > 0.5).int() == y_true
                ).float().sum() / eval_size
        is_normal = is_normal.detach().cpu().numpy()  # (T, 1)
        if self.config["loss_type"] == "GreedyLoss":
            # (T, n_eval_split)
            y_outliter_preds = torch.cat(y_outliter_preds, dim=1).detach().cpu().numpy()
            # (T, n_eval_split)
            y_anomaly_preds = torch.cat(y_anomaly_preds, dim=1).detach().cpu().numpy()
            y_preds = alpha * y_outliter_preds + (1 - alpha) * y_anomaly_preds
            # check directory
            if len(save_csv) == 0:
                dirname = os.path.join(
                    self.config["outdir"], f"predictions/{self.steps}steps"
                )
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                df = pd.DataFrame(y_outliter_preds, columns=keys_list)
                df.to_csv(
                    os.path.join(dirname, f"preds_outliter_{self.steps}.csv"),
                    index=False,
                )
                df = pd.DataFrame(y_anomaly_preds, columns=keys_list)
                df.to_csv(
                    os.path.join(dirname, f"preds_anomaly_{self.steps}.csv"),
                    index=False,
                )
        else:
            y_preds = (
                torch.cat(y_preds, dim=1).detach().cpu().numpy()
            )  # (T, n_eval_split)
            # check directory
            if len(save_csv) == 0:
                dirname = os.path.join(
                    self.config["outdir"], f"predictions/{self.steps}steps"
                )
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
        if (len(save_csv) != 0) and (
            save_csv
            in [
                "dev",
                "eval",
                "00",
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
            ]
        ):
            dirname = os.path.join(
                self.config["outdir"],
                f"split{self.config['n_eval_split']}frame{self.config['max_frames']}",
            )
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        df = pd.DataFrame(y_preds, columns=keys_list)
        df.to_csv(os.path.join(dirname, f"preds_{self.steps}.csv"), index=False)
        self.eval_asd(
            is_normal=is_normal,
            y_preds=y_preds,
            dirname=dirname,
            recode=self.train,
        )
        # restore mode
        self.model.train()

    def eval_asd(self, is_normal, y_preds, dirname, recode=True):
        df = pd.DataFrame(np.array(is_normal), columns=["is_normal"])
        df.to_csv(os.path.join(dirname, "is_normal.csv"), index=False)

        columns = ["min", "max", "median", "mean", "median_mean"]
        n_cut = max(self.n_eval_split // 5, 1)
        y_scores = [
            y_preds.min(axis=1),
            y_preds.max(axis=1),
            y_preds.mean(axis=1),
            np.median(y_preds, axis=1),
            np.sort(y_preds, axis=1)[:, n_cut:-n_cut].mean(axis=1),
        ]
        df = pd.DataFrame(np.array(y_scores).T, columns=columns)
        df.to_csv(os.path.join(dirname, "scores.csv"), index=False)
        results = []
        df_columns = []
        plt.figure()
        best_col = ""
        best_AUC = 0
        best_pAUC = 0
        for i, column in enumerate(columns):
            y_score = y_scores[i]
            # calculate roc curve
            try:
                self.eval_metric[f"eval_metric/AUC_{column}"] = roc_auc_score(
                    y_true=is_normal, y_score=y_score
                )
                self.eval_metric[f"eval_metric/pAUC_{column}"] = roc_auc_score(
                    y_true=is_normal, y_score=y_score, max_fpr=0.1
                )
                fpr, tpr, thresholds = roc_curve(y_true=is_normal, y_score=y_score)
                if recode:
                    # recode
                    self._write_to_tensorboard(self.eval_metric)
                plt.plot(fpr, tpr, label=column)
            except ValueError:
                logging.warning("Raise ValueError: May be contain NaN in y_pred.")
                pass

            df_columns.append("AUC_" + column)
            df_columns.append("pAUC_" + column)
            results.append(self.eval_metric[f"eval_metric/AUC_{column}"])
            results.append(self.eval_metric[f"eval_metric/pAUC_{column}"])
            if best_AUC < self.eval_metric[f"eval_metric/AUC_{column}"]:
                best_AUC = self.eval_metric[f"eval_metric/AUC_{column}"]
                best_col = column
            if best_pAUC < self.eval_metric[f"eval_metric/pAUC_{column}"]:
                best_pAUC = self.eval_metric[f"eval_metric/pAUC_{column}"]
        plt.xlabel("FPR: False positive rate")
        plt.ylabel("TPR: True positive rate")
        plt.grid()
        plt.title(
            f"ROC curve @ {self.steps} steps\n"
            + f"{best_col} AUC:{best_AUC:.4f}, pAUC:{best_pAUC:.4f}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(dirname, f"roc_curve_{self.steps}.png"))
        plt.close()

        df = pd.DataFrame([results], columns=df_columns)
        df.to_csv(os.path.join(dirname, "results.csv"), index=False)
        for key in self.eval_metric.keys():
            logging.info(f"(Steps: {self.steps}) {key} = {self.eval_metric[key]:.4f}.")
        # reset
        self.eval_metric = defaultdict(float)

    def plot_embedding(self, embedding, label, name="", dirname=""):
        """Plot distribution of embedding layer.

        Args:
            embedding (ndarray): (B, 2048)
            label (ndarray): (B,)
        """
        X_embedded = self.tsne.fit_transform(embedding)
        colors = ["r", "g", "b"]
        label_names = ["normal", "anomaly", "outliter"]
        plt.figure()
        for i, label_name in enumerate(label_names):
            tmp = X_embedded[label == label_name]
            plt.scatter(
                tmp[:, 0], tmp[:, 1], alpha=0.3, label=label_name, color=colors[i]
            )
        plt.legend()
        plt.title(f"Distribution of {name} embedding layer at {self.epochs}.")
        if len(dirname) == 0:
            dirname = os.path.join(self.config["outdir"], "predictions", "embedding")
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        plt.savefig(os.path.join(dirname, f"{name}epoch{self.epochs}.png"))

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl")
            )
            self.eval_score(save_csv="")
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                # self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(
                    f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}."
                )
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True
