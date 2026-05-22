import torch
import torchmetrics

from wsi_classification.experiments.default_cfg import ExperimentConfig
from wsi_classification.experiments.lightning_wrappers.base_lightning_wrapper import LightningWrapperBase


class MILWrapper(LightningWrapperBase):
    """Lightning wrapper for Multiple Instance Learning (MIL) classification tasks."""

    def __init__(
        self,
        network: torch.nn.Module,
        cfg: ExperimentConfig,
        use_bce_loss: bool = True,
    ):
        """Initialize the MILWrapper.

        Args:
            network: MIL network to wrap. Must expose an ``out_features`` attribute
                when used for multiclass classification.
            cfg: Experiment configuration.
            use_bce_loss: Use BCEWithLogitsLoss for binary classification.
                When *network.out_features* > 1 and this is False, CrossEntropyLoss
                is used instead.
        """
        super().__init__(network=network, cfg=cfg)

        self.multiclass = hasattr(network, "out_features") and network.out_features > 1

        if self.multiclass:
            acc_kwargs = {"task": "multiclass", "num_classes": network.out_features}
        else:
            acc_kwargs = {"task": "binary"}

        self.train_acc = torchmetrics.Accuracy(**acc_kwargs)
        self.val_acc = torchmetrics.Accuracy(**acc_kwargs)
        self.test_acc = torchmetrics.Accuracy(**acc_kwargs)

        # AUROC, AUPRC and F1 Score metrics
        self.val_auroc = torchmetrics.AUROC(**acc_kwargs)
        self.val_auprc = torchmetrics.AveragePrecision(**acc_kwargs)
        self.val_f1 = torchmetrics.F1Score(**acc_kwargs)
        self.test_auroc = torchmetrics.AUROC(**acc_kwargs)
        self.test_auprc = torchmetrics.AveragePrecision(**acc_kwargs)
        self.test_f1 = torchmetrics.F1Score(**acc_kwargs)

        self.use_bce_loss = use_bce_loss
        if self.multiclass and not self.use_bce_loss:
            self.loss_metric = torch.nn.CrossEntropyLoss()
        else:
            self.loss_metric = torch.nn.BCEWithLogitsLoss()

    def _step(
        self, batch: dict[str, torch.Tensor], accuracy_calculator: torchmetrics.Metric
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Shared forward + loss computation for train and validation.

        Args:
            batch: Dict with keys ``"input"`` (B, N, D), ``"label"`` (B,), and optionally ``"coords"`` (B, N, 2).
            accuracy_calculator: Metric accumulator to update with this step's predictions.

        Returns:
            A 3-tuple of ``(loss, predictions, output_dict)``.
        """
        inputs = batch["input"]
        labels = batch["label"]

        # Pass coords to model if it supports them (for spatial models)
        if getattr(self.network, "uses_coords", False) and "coords" in batch:
            output_dict = self.network(inputs, coords=batch["coords"])
        else:
            output_dict = self.network(inputs)
        logits = output_dict["logits"].squeeze(1)

        if not self.multiclass and self.use_bce_loss:
            loss = self.loss_metric(logits, labels.float())
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int()
        else:
            loss = self.loss_metric(logits, labels)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

        accuracy_calculator.update(preds, labels)
        return loss, preds, {"logits": logits, "probs": probs}

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform one training step and log loss.

        Args:
            batch: Dict with ``"input"`` and ``"label"`` tensors.
            batch_idx: Index of the current batch (unused).

        Returns:
            Scalar training loss.
        """
        loss, _, _ = self._step(batch, self.train_acc)
        self.log(
            "train/loss", loss,
            on_step=True, on_epoch=True, sync_dist=True,
            batch_size=batch["input"].size(0),
        )
        return loss

    def on_train_epoch_end(self) -> None:
        """Log epoch-level training accuracy and reset the accumulator."""
        acc = self.train_acc.compute()
        self.log("train/acc", acc, sync_dist=True)
        self.train_acc.reset()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform one validation step and log loss.

        Args:
            batch: Dict with ``"input"`` and ``"label"`` tensors.
            batch_idx: Index of the current batch (unused).

        Returns:
            Scalar validation loss.
        """
        loss, preds, output_dict = self._step(batch, self.val_acc)
        probs = output_dict["probs"]
        labels = batch["label"]

        # Update AUROC, AUPRC and F1 metrics
        self.val_auroc.update(probs, labels)
        self.val_auprc.update(probs, labels)
        self.val_f1.update(preds, labels)

        self.log(
            "val/loss", loss,
            on_step=False, on_epoch=True, sync_dist=True,
            batch_size=batch["input"].size(0),
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        """Log epoch-level validation accuracy, AUROC, AUPRC, F1 and reset the accumulators."""
        acc = self.val_acc.compute()
        auroc = self.val_auroc.compute()
        auprc = self.val_auprc.compute()
        f1 = self.val_f1.compute()
        self.log("val/acc", acc, sync_dist=True)
        self.log("val/auroc", auroc, sync_dist=True)
        self.log("val/auprc", auprc, sync_dist=True)
        self.log("val/f1", f1, sync_dist=True)
        self.val_acc.reset()
        self.val_auroc.reset()
        self.val_auprc.reset()
        self.val_f1.reset()

    def on_fit_end(self) -> None:
        """Final log at end of training to ensure best validation metrics are captured."""
        if self.logger is not None:
            final_metrics = {
                "final/val_acc": self.trainer.callback_metrics.get("val/acc"),
                "final/val_auroc": self.trainer.callback_metrics.get("val/auroc"),
                "final/val_auprc": self.trainer.callback_metrics.get("val/auprc"),
                "final/val_f1": self.trainer.callback_metrics.get("val/f1"),
            }
            self.logger.experiment.log(final_metrics)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform one test step and log predictions with slide names.

        Args:
            batch: Dict with ``"input"``, ``"label"``, and ``"slide_name"`` tensors.
            batch_idx: Index of the current batch (unused).
        """
        inputs = batch["input"]
        labels = batch["label"]
        slide_names = batch["slide_name"]

        # Pass coords to model if it supports them (for spatial models)
        if getattr(self.network, "uses_coords", False) and "coords" in batch:
            output_dict = self.network(inputs, coords=batch["coords"])
        else:
            output_dict = self.network(inputs)
        logits = output_dict["logits"].squeeze(1)

        if not self.multiclass and self.use_bce_loss:
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int()
        else:
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

        # Update metrics
        self.test_acc.update(preds, labels)
        self.test_auroc.update(probs, labels)
        self.test_auprc.update(probs, labels)
        self.test_f1.update(preds, labels)

        # Log predictions per slide
        for slide_name, pred, prob, label in zip(slide_names, preds, probs, labels):
            self.log_dict(
                {
                    f"test/pred_{slide_name}": pred.item(),
                    f"test/prob_{slide_name}": prob.max().item() if isinstance(prob, torch.Tensor) else prob,
                    f"test/label_{slide_name}": label.item(),
                },
                sync_dist=True,
            )

    def on_test_epoch_end(self) -> None:
        """Log epoch-level test accuracy, AUROC, AUPRC, F1 and reset the accumulators."""
        acc = self.test_acc.compute()
        auroc = self.test_auroc.compute()
        auprc = self.test_auprc.compute()
        f1 = self.test_f1.compute()
        self.log("test/acc", acc, sync_dist=True)
        self.log("test/auroc", auroc, sync_dist=True)
        self.log("test/auprc", auprc, sync_dist=True)
        self.log("test/f1", f1, sync_dist=True)
        self.test_acc.reset()
        self.test_auroc.reset()
        self.test_auprc.reset()
        self.test_f1.reset()

    def on_test_end(self) -> None:
        """Final log at end of testing to ensure metrics are captured."""
        if self.logger is not None:
            final_metrics = {
                "final/test_acc": self.trainer.callback_metrics.get("test/acc"),
                "final/test_auroc": self.trainer.callback_metrics.get("test/auroc"),
                "final/test_auprc": self.trainer.callback_metrics.get("test/auprc"),
                "final/test_f1": self.trainer.callback_metrics.get("test/f1"),
            }
            self.logger.experiment.log(final_metrics)
