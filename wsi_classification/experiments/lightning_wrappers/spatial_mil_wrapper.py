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
            preds = (torch.sigmoid(logits) >= 0.5).int()
        else:
            loss = self.loss_metric(logits, labels)
            preds = torch.argmax(logits, dim=-1)

        accuracy_calculator.update(preds, labels)
        return loss, preds, {"logits": logits}
