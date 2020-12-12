import logging
import os
import sys
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from tensorboardX import SummaryWriter

sys.path.append("../../")
sys.path.append("../input/modules")
from losses import CenterLoss  # noqa: E402
from utils import lwlrap  # noqa: E402


class SEDTrainer(object):
    """Customized trainer module for SED training."""

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
        l_spec=5626,
        save_name="",
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
        self.l_spec = l_spec
        if train:
            self.writer = SummaryWriter(config["outdir"])
        self.use_center_loss = use_center_loss
        self.save_name = save_name
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
        self.best_score = 0
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
        print(y_)
        print(x.shape)
        if not torch.isnan(loss):
            loss = loss / self.config["accum_grads"]
            print(f"loss:{loss.item()}")
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
            self._check_log_interval()
            self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return
        try:
            self.epoch_train_loss["train/epoch_lwlrap"] = lwlrap(
                y_true=self.train_y_epoch, y_score=self.train_pred_epoch
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
        self.epoch_train_loss = defaultdict(float)
        if self.use_center_loss:
            self.train_embedding_epoch = np.empty(
                (0, self.config["center_loss_params"]["feat_dim"])
            )
            self.train_label_epoch = np.empty((0, 1))

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        x = batch["X"].to(self.device)
        y_frame = batch["y_frame"].to(self.device)
        y_clip = batch["y_clip"].to(self.device)
        batch_size = x.size(0)
        y_ = self.model(x)
        if self.config["loss_type"] == "FrameClipLoss":
            loss = self.criterion(y_["y_frame"], y_frame, y_["y_clip"], y_clip)
        if self.use_center_loss:
            center_loss_label = batch["label"]
            loss += (
                self.center_loss(y_["embedding"], center_loss_label)
                * self.config["center_loss_alpha"]
            )
        # add to total eval loss
        self.total_eval_loss["dev/loss"] += loss.item() / batch_size
        if self.config["loss_type"] == "FrameClipLoss":
            self.dev_pred_epoch = np.concatenate(
                [self.dev_pred_epoch, y_["y_clip"].detach().cpu().numpy()], axis=0
            )
            self.dev_y_epoch = np.concatenate(
                [self.dev_y_epoch, y_clip.detach().cpu().numpy()], axis=0
            )
        if self.use_center_loss:
            self.dev_label_epoch = np.concatenate(
                [self.dev_label_epoch, center_loss_label], axis=0
            )
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
            self.epoch_dev_loss["dev/epoch_lwlrap"] = lwlrap(
                y_true=self.dev_y_epoch, y_score=self.dev_pred_epoch
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
        if self.epochs % self.config["eval_interval_epochs"] == 0:
            items = self.inference(mode="valid")
            logging.info(
                f"Inference (Epochs: {self.epochs}) lwlrap: {items['score']:.6f}"
            )
            if items["score"] > self.best_score:
                self.best_score = items["score"]
                logging.info(
                    f"Epochs: {self.epochs}, BEST score was updated {self.best_score:.6f}."
                )
                self.save_checkpoint(
                    os.path.join(
                        self.config["outdir"], f"best_score{self.save_name}.pkl"
                    )
                )
                logging.info(f"Best model was updated @ {self.steps} steps.")
            self._write_to_tensorboard(self.eval_metric)
            self.eval_metric = defaultdict(float)
        # record
        self._write_to_tensorboard(self.total_eval_loss)
        self._write_to_tensorboard(self.epoch_eval_loss)
        if self.use_center_loss and (
            self.epochs % self.config["eval_interval_epochs"] == 0
        ):
            self.plot_embedding(
                self.dev_embedding_epoch, self.dev_label_epoch, name="dev"
            )

        # reset
        self.total_eval_loss = defaultdict(float)
        self.epoch_eval_loss = defaultdict(float)
        self.dev_pred_epoch = np.empty((0, 1))
        self.dev_y_epoch = np.empty((0, 1))
        if self.use_center_loss:
            self.dev_embedding_epoch = np.empty(
                (0, self.config["center_loss_params"]["feat_dim"])
            )
            self.dev_label_epoch = np.empty((0, 1))
        # restore mode
        self.model.train()

    def inference(self, mode="test"):
        """Evaluate and save intermediate result."""
        # evaluate
        keys_list = [f"X{i}" for i in range(self.n_eval_split)]
        y_clip = [
            torch.empty((0, self.config["n_class"])).to(self.device)
            for _ in range(self.n_eval_split)
        ]
        y_frame = [
            torch.empty((0, self.l_spec, self.config["n_class"])).to(self.device)
            for _ in range(self.n_eval_split)
        ]
        y_clip_true = torch.empty((0, self.config["n_class"]))
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.data_loader["eval"]):
                if mode == "valid":
                    y_clip_true = torch.cat([y_clip_true, batch["y_clip"]], dim=0)
                x_batchs = [batch[key].to(self.device) for key in keys_list]
                for i in range(self.n_eval_split):
                    y_batch_ = self.model(x_batchs[i])
                    y_clip[i] = torch.cat([y_clip[i], y_batch_["y_clip"]], dim=0)
                    y_frame[i] = torch.cat([y_frame[i], y_batch_["y_frame"]], dim=0)
        # (B, n_eval_split, n_class)
        y_clip = torch.cat(y_clip, dim=1).detach().cpu().numpy()
        # (B, n_eval_split, T, n_class)
        y_frame = torch.cat(y_frame, dim=1).detach().cpu().numpy()
        if mode == "valid":
            y_clip_true = y_clip_true.numpy()
            score = lwlrap(y_clip_true, y_clip.max(axis=1))
            self.eval_metric["eval_metric/lwlrap"] = score
            return {"y_clip": y_clip, "y_frame": y_frame, "score": score}
        return {"y_clip": y_clip, "y_frame": y_frame}

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
                os.path.join(
                    self.config["outdir"],
                    f"checkpoint-{self.steps}steps{self.save_name}.pkl",
                )
            )
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
