import torch
import torchmetrics

from wsi_classification.experiments.lightning_wrappers.mil_wrapper import MILWrapper


class SpatialMILWrapper(MILWrapper):
    """MIL wrapper that forwards tile coordinates to spatially-aware models.

    Identical to :class:`MILWrapper` except that ``batch["coords"]`` is extracted
    and passed as a second positional argument to the network.  Use this wrapper
    with models whose ``forward`` signature is ``forward(x, coords)``.
    """

    def _step(self, batch, accuracy_calculator):
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

        accuracy_calculator.update(preds, labels)
        return loss, preds, {"logits": logits, "probs": probs}

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
