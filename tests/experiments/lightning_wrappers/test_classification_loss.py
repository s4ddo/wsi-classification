"""Tests for the classification loss and wrapper functionality."""

import warnings

import torch

from wsi_classification.experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
from wsi_classification.experiments.default_cfg import ExperimentConfig


# Mock Network
class MockNet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.out_proj = torch.nn.Linear(10, num_classes)

    def forward(self, input_and_condition):
        # Extract input from the dictionary passed by ClassificationWrapper
        x = input_and_condition["input"]
        return {"logits": self.out_proj(x)}


def test_bce_loss():
    """Test that BCE loss is correctly configured when use_bce_loss=True."""
    cfg = ExperimentConfig()
    net = MockNet(num_classes=10)

    # Test 1: Initialize with use_bce_loss=True
    wrapper_bce = ClassificationWrapper(net, cfg, use_bce_loss=True)
    assert isinstance(wrapper_bce.loss_metric, torch.nn.BCEWithLogitsLoss), \
        "Expected BCEWithLogitsLoss when use_bce_loss=True"

    # Test 2: Initialize with use_bce_loss=False (Default)
    wrapper_ce = ClassificationWrapper(net, cfg, use_bce_loss=False)
    assert isinstance(wrapper_ce.loss_metric, torch.nn.CrossEntropyLoss), \
        "Expected CrossEntropyLoss when use_bce_loss=False"


def test_binary_classification():
    """Test that binary classification uses BCEWithLogitsLoss."""
    cfg = ExperimentConfig()
    net = MockNet(num_classes=1)

    wrapper = ClassificationWrapper(net, cfg)
    assert isinstance(wrapper.loss_metric, torch.nn.BCEWithLogitsLoss), \
        "Expected BCEWithLogitsLoss for binary classification"
    assert wrapper.multiclass is False


def test_multiclass_classification():
    """Test that multiclass classification uses CrossEntropyLoss by default."""
    cfg = ExperimentConfig()
    net = MockNet(num_classes=10)

    wrapper = ClassificationWrapper(net, cfg)
    assert isinstance(wrapper.loss_metric, torch.nn.CrossEntropyLoss), \
        "Expected CrossEntropyLoss for multiclass classification"
    assert wrapper.multiclass is True


def test_prediction_methods():
    """Test the prediction methods for multiclass and binary classification."""
    # Multiclass
    logits_multi = torch.tensor([[0.1, 0.9, 0.3], [0.8, 0.1, 0.1]])
    preds_multi = ClassificationWrapper.multiclass_prediction(logits_multi)
    assert preds_multi.tolist() == [1, 0]

    # Binary
    logits_binary = torch.tensor([[0.5], [-0.5]])
    preds_binary = ClassificationWrapper.binary_prediction(logits_binary)
    assert preds_binary.tolist() == [1, 0]


def test_validation_step():
    """Test that validation_step is properly configured."""
    cfg = ExperimentConfig()
    net = MockNet(num_classes=10)
    wrapper = ClassificationWrapper(net, cfg)

    # Create a mock batch with proper structure
    batch = {
        "input": torch.randn(2, 10),
        "label": torch.tensor([1, 5], dtype=torch.long),
        "condition": {}
    }

    # Mock the log and distributed settings
    wrapper.distributed = False
    logged_values = {}
    def mock_log(name, value, **kwargs):
        logged_values[name] = value
    wrapper.log = mock_log

    # Call validation step
    loss = wrapper.validation_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor), "Validation step must return a tensor loss"
    assert loss.ndim == 0, "Loss must be a scalar"
    assert torch.isfinite(loss), "Loss must be finite"
    assert "val/loss" in logged_values, "val/loss should be logged"


def test_validation_state_tracking():
    """Test that validation state is properly tracked in the wrapper."""
    cfg = ExperimentConfig()
    net = MockNet(num_classes=2)
    wrapper = ClassificationWrapper(net, cfg)

    # Check that validation metrics are initialized
    assert hasattr(wrapper, 'val_acc'), "Wrapper must have val_acc metric"
    assert hasattr(wrapper, 'best_val_acc'), "Wrapper must have best_val_acc tracking"
    assert hasattr(wrapper, 'best_val_loss'), "Wrapper must have best_val_loss tracking"

    # Check that other_outputs_validation list is initialized
    assert hasattr(wrapper, 'other_outputs_validation'), "Wrapper must have other_outputs_validation list"
    assert isinstance(wrapper.other_outputs_validation, list), "other_outputs_validation must be a list"

    # Simulate some validation outputs
    wrapper.other_outputs_validation.append({'logits': torch.randn(2, 2)})
    wrapper.other_outputs_validation.append({'logits': torch.randn(2, 2)})
    assert len(wrapper.other_outputs_validation) == 2, "Validation outputs should be accumulated"

    # Manually simulate what on_validation_epoch_end does: clear the list
    wrapper.other_outputs_validation.clear()
    assert len(wrapper.other_outputs_validation) == 0, "Validation outputs should be cleared"
