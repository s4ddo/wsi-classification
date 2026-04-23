import pytest
import torch
from wsi_classification.models.clam import CLAM_SB, CLAM_MB, CLAM


class TestCLAMSB:
    """Test suite for CLAM_SB (Single-Branch)."""

    def test_initialization(self):
        """Test CLAM_SB initialization with default parameters."""
        model = CLAM_SB()
        assert isinstance(model, CLAM_SB)
        assert model.n_classes == 2
        assert model.gate is True
        assert model.size_arg == "small"

    def test_initialization_big_model(self):
        """Test CLAM_SB initialization with big model size."""
        model = CLAM_SB(size_arg="big", n_classes=3)
        assert model.size_arg == "big"
        assert model.n_classes == 3

    def test_forward_pass_2d(self):
        """Test forward pass with 2D input (N, D)."""
        model = CLAM_SB(n_classes=2, embed_dim=1024)
        x = torch.randn(100, 1024)  # 100 patches, 1024-dim features

        out = model(x)

        assert isinstance(out, dict)
        assert "logits" in out
        assert "probs" in out
        assert "pred" in out
        assert "attention" in out
        # Binary classification: logits should have 2 classes
        assert out["logits"].shape == torch.Size([2])

    def test_forward_pass_3d(self):
        """Test forward pass with 3D input (B, N, D)."""
        model = CLAM_SB(n_classes=2, embed_dim=512)
        x = torch.randn(1, 80, 512)  # Batch of 1, 80 patches

        out = model(x)

        assert out["logits"].shape == torch.Size([2])

    def test_multiclass(self):
        """Test with multi-class classification."""
        model = CLAM_SB(n_classes=4, embed_dim=1024)
        x = torch.randn(150, 1024)

        out = model(x)

        # For multi-class, logits should have n_classes dimensions
        assert out["logits"].shape == torch.Size([4])

    def test_instance_eval(self):
        """Test with instance_eval=True."""
        model = CLAM_SB(n_classes=2, embed_dim=512)
        x = torch.randn(50, 512)

        out = model(x, instance_eval=True)

        assert "instance_logits" in out
        # Shape: (N, n_classes, 2) for binary instance classification
        assert out["instance_logits"].shape == torch.Size([50, 2, 2])

    def test_return_features(self):
        """Test return_features flag."""
        model = CLAM_SB(n_classes=2, embed_dim=1024)
        x = torch.randn(100, 1024)

        out = model(x, return_features=True)

        assert "features" in out
        # Features should be aggregated representation (1, hidden[1])
        assert out["features"].shape == torch.Size([1, 512])  # hidden[1] for small model

    def test_attention_only(self):
        """Test attention_only flag."""
        model = CLAM_SB(embed_dim=512)
        x = torch.randn(80, 512)

        out = model(x, attention_only=True)

        assert "attention" in out
        # Only attention should be returned
        assert len(out) == 1
        # Attention shape: (1, N)
        assert out["attention"].shape == torch.Size([1, 80])

    def test_dropout_effect(self):
        """Test that dropout parameter is used."""
        model_with_dropout = CLAM_SB(dropout=0.5, n_classes=2, embed_dim=512)
        model_no_dropout = CLAM_SB(dropout=0.0, n_classes=2, embed_dim=512)

        # Check dropout is configured (at index 2 if added)
        has_dropout = any(
            isinstance(m, torch.nn.Dropout) and m.p == 0.5
            for m in model_with_dropout.feature_extractor.modules()
        )
        assert has_dropout

    def test_ungated_attention(self):
        """Test with gated=False."""
        model = CLAM_SB(gate=False, embed_dim=512, n_classes=2)
        x = torch.randn(100, 512)

        out = model(x)

        assert "logits" in out
        assert out["logits"].shape == torch.Size([2])

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = CLAM_SB(n_classes=2, embed_dim=512)
        x = torch.randn(50, 512, requires_grad=True)

        out = model(x)
        loss = out["logits"].sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestCLAMMB:
    """Test suite for CLAM_MB (Multi-Branch)."""

    def test_initialization(self):
        """Test CLAM_MB initialization."""
        model = CLAM_MB(n_classes=3, embed_dim=1024)
        assert isinstance(model, CLAM_MB)
        assert model.n_classes == 3

    def test_forward_pass_2d(self):
        """Test forward pass with 2D input."""
        model = CLAM_MB(n_classes=3, embed_dim=1024)
        x = torch.randn(100, 1024)

        out = model(x)

        assert "logits" in out
        assert "attention" in out
        assert out["logits"].shape == torch.Size([3])
        # Multi-branch attention: (n_classes, N)
        assert out["attention"].shape == torch.Size([3, 100])

    def test_multiclass(self):
        """Test with multiple classes."""
        model = CLAM_MB(n_classes=5, embed_dim=512)
        x = torch.randn(150, 512)

        out = model(x)

        assert out["logits"].shape == torch.Size([5])
        assert out["attention"].shape == torch.Size([5, 150])

    def test_class_specific_attention(self):
        """Test that each class has specific attention weights."""
        model = CLAM_MB(n_classes=3, embed_dim=512)
        x = torch.randn(50, 512)

        out = model(x)

        attention = out["attention"]
        # Each class should have different attention weights
        assert attention.shape == torch.Size([3, 50])

        # Verify each class's attention sums to 1
        for i in range(3):
            assert torch.isclose(attention[i].sum(), torch.tensor(1.0), atol=1e-5)

    def test_instance_eval(self):
        """Test with instance_eval=True."""
        model = CLAM_MB(n_classes=3, embed_dim=512)
        x = torch.randn(50, 512)

        out = model(x, instance_eval=True)

        assert "instance_logits" in out
        assert out["instance_logits"].shape == torch.Size([50, 3, 2])

    def test_return_features(self):
        """Test return_features flag."""
        model = CLAM_MB(n_classes=3, embed_dim=1024)
        x = torch.randn(100, 1024)

        out = model(x, return_features=True)

        assert "features" in out
        # Features should be mean of class-specific aggregations (hidden[1],)
        assert out["features"].shape == torch.Size([512])  # hidden[1] for small model


class TestCLAMAlias:
    """Test that CLAM is an alias for CLAM_SB."""

    def test_clam_alias(self):
        """Test that CLAM == CLAM_SB."""
        assert CLAM is CLAM_SB

    def test_clam_creates_sb(self):
        """Test that creating CLAM creates CLAM_SB instance."""
        model = CLAM(n_classes=2, embed_dim=512)
        assert isinstance(model, CLAM_SB)
