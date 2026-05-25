import torch
import torchmetrics

from wsi_classification.experiments.lightning_wrappers.mil_wrapper import MILWrapper


class SpatialMILWrapper(MILWrapper):
    """MIL wrapper that forwards tile coordinates to spatially-aware models.

    Identical to :class:`MILWrapper` except that ``batch["coords"]`` is extracted
    and passed as a second positional argument to the network.  Use this wrapper
    with models whose ``forward`` signature is ``forward(x, coords)``.

    Supports SlideMoE models that return auxiliary losses (aux_logits and load_balance_loss).
    """

    def __init__(
        self,
        network: torch.nn.Module,
        cfg,
        use_bce_loss: bool = True,
        aux_scorer_coef: float = 0.1,
        load_balance_coef: float = 0.01,
    ):
        """Initialize the SpatialMILWrapper.

        Args:
            network: MIL network to wrap.
            cfg: Experiment configuration.
            use_bce_loss: Use BCEWithLogitsLoss for binary classification.
            aux_scorer_coef: Coefficient for auxiliary scorer head loss (SlideMoE).
            load_balance_coef: Coefficient for MoE load balancing loss (SlideMoE).
        """
        super().__init__(network=network, cfg=cfg, use_bce_loss=use_bce_loss)
        self.aux_scorer_coef = aux_scorer_coef
        self.load_balance_coef = load_balance_coef

    def _step(self, batch, accuracy_calculator, use_aux_losses: bool = False):
        inputs = batch["input"]
        labels = batch["label"]

        coords = batch.get("coords")
        if isinstance(coords, list):
            coords = torch.stack(coords, dim=0)

        output_dict = self.network(inputs, coords)
        logits = output_dict["logits"].squeeze(1)

        if not self.multiclass and self.use_bce_loss:
            loss = self.loss_metric(logits, labels.float())
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int()
        else:
            loss = self.loss_metric(logits, labels)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

        # Add auxiliary losses if available and requested (training only)
        aux_losses = {}
        if use_aux_losses:
            # Auxiliary scorer head loss
            if "aux_logits" in output_dict and output_dict["aux_logits"] is not None:
                aux_logits = output_dict["aux_logits"].squeeze(1)
                if not self.multiclass and self.use_bce_loss:
                    loss_aux = self.loss_metric(aux_logits, labels.float())
                else:
                    loss_aux = self.loss_metric(aux_logits, labels)
                loss = loss + self.aux_scorer_coef * loss_aux
                aux_losses["aux_loss"] = loss_aux.detach()

            # MoE load balancing loss
            if "load_balance_loss" in output_dict:
                loss_lb = output_dict["load_balance_loss"]
                loss = loss + self.load_balance_coef * loss_lb
                aux_losses["load_balance_loss"] = loss_lb.detach()

        accuracy_calculator.update(preds, labels)
        return loss, preds, {"logits": logits, "probs": probs, **aux_losses}

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform one training step with auxiliary losses for SlideMoE.

        Args:
            batch: Dict with ``"input"`` and ``"label"`` tensors.
            batch_idx: Index of the current batch (unused).

        Returns:
            Scalar training loss.
        """
        loss, _, aux_outputs = self._step(batch, self.train_acc, use_aux_losses=True)
        self.log(
            "train/loss", loss,
            on_step=True, on_epoch=True, sync_dist=True,
            batch_size=batch["input"].size(0),
        )
        # Log auxiliary losses if present
        if "aux_loss" in aux_outputs:
            self.log(
                "train/aux_loss", aux_outputs["aux_loss"],
                on_step=True, on_epoch=True, sync_dist=True,
                batch_size=batch["input"].size(0),
            )
        if "load_balance_loss" in aux_outputs:
            self.log(
                "train/load_balance_loss", aux_outputs["load_balance_loss"],
                on_step=True, on_epoch=True, sync_dist=True,
                batch_size=batch["input"].size(0),
            )
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """Perform one test step with spatial coordinates.

        Args:
            batch: Dict with ``"input"``, ``"label"``, ``"coords"``, and ``"slide_name"``.
            batch_idx: Index of the current batch (unused).
        """
        inputs = batch["input"]
        labels = batch["label"]
        slide_names = batch["slide_name"]

        coords = batch.get("coords")
        if isinstance(coords, list):
            coords = torch.stack(coords, dim=0)

        output_dict = self.network(inputs, coords)
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
