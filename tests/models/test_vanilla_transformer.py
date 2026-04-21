import pytest
import torch
from wsi_classification.models.vanilla_transformer import VanillaTransformer


def test_vanilla_transformer_initialization():
    """Test that VanillaTransformer initializes correctly with default params."""
    model = VanillaTransformer(
        in_features=1280, hidden_dim=2048, out_features=2, num_heads=8, num_layers=4
    )
    assert isinstance(model, VanillaTransformer)
    assert hasattr(model, "transformer_encoder")
    assert hasattr(model, "classifier")
    assert hasattr(model, "cls_token")


def test_vanilla_transformer_forward_pass_batch():
    """Test forward pass with batched input (B, N, D)."""
    model = VanillaTransformer(
        in_features=1280, hidden_dim=512, out_features=2, num_heads=8, num_layers=2
    )
    x = torch.randn(2, 50, 1280)  # Batch of 2, 50 patches, 1280 dim

    out = model(x)

    assert isinstance(out, dict)
    assert "logits" in out
    assert out["logits"].shape == (2, 2)


def test_vanilla_transformer_forward_pass_unbatched():
    """Test forward pass with unbatched input (N, D)."""
    model = VanillaTransformer(in_features=1280, hidden_dim=512, out_features=1, num_layers=2)
    x = torch.randn(100, 1280)  # Single sequence, 100 patches, 1280 dim

    out = model(x)

    assert isinstance(out, dict)
    assert "logits" in out
    # Since it adds a batch dimension dynamically, output should be (1, 1)
    assert out["logits"].shape == (1, 1)


def test_vanilla_transformer_attention_return():
    """Test return_attention flag returns simplified attention scores."""
    model = VanillaTransformer(in_features=1280, out_features=1, num_layers=2)
    x = torch.randn(4, 75, 1280)  # Batch of 4, 75 patches, 1280 dim

    out = model(x, return_attention=True)

    assert isinstance(out, dict)
    assert "logits" in out
    assert "attention" in out

    assert out["logits"].shape == (4, 1)

    # Check attention shape is (B, 1, N+1) for CLS pooling (includes CLS token)
    attn = out["attention"]
    assert attn.shape == (4, 1, 76)  # 75 patches + 1 CLS token

    # Check if attention weights sum to 1 over the sequence dimension
    attn_sum = attn.sum(dim=2)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)


def test_vanilla_transformer_pool_methods():
    """Test different pooling methods: 'cls', 'mean', 'max'."""
    x = torch.randn(2, 40, 1280)

    # Test CLS pooling
    model_cls = VanillaTransformer(
        in_features=1280, out_features=2, pool_method="cls", num_layers=1
    )
    out_cls = model_cls(x)
    assert out_cls["logits"].shape == (2, 2)

    # Test mean pooling
    model_mean = VanillaTransformer(
        in_features=1280, out_features=2, pool_method="mean", num_layers=1
    )
    out_mean = model_mean(x)
    assert out_mean["logits"].shape == (2, 2)

    # Test max pooling
    model_max = VanillaTransformer(
        in_features=1280, out_features=2, pool_method="max", num_layers=1
    )
    out_max = model_max(x)
    assert out_max["logits"].shape == (2, 2)


def test_vanilla_transformer_pool_attention_shapes():
    """Test that attention shapes are correct for different pooling methods."""
    x = torch.randn(2, 50, 1280)

    # CLS pooling includes CLS token
    model_cls = VanillaTransformer(in_features=1280, pool_method="cls", num_layers=1)
    out_cls = model_cls(x, return_attention=True)
    assert out_cls["attention"].shape == (2, 1, 51)

    # Mean and max pooling don't add CLS token
    model_mean = VanillaTransformer(in_features=1280, pool_method="mean", num_layers=1)
    out_mean = model_mean(x, return_attention=True)
    assert out_mean["attention"].shape == (2, 1, 50)

    model_max = VanillaTransformer(in_features=1280, pool_method="max", num_layers=1)
    out_max = model_max(x, return_attention=True)
    assert out_max["attention"].shape == (2, 1, 50)


def test_vanilla_transformer_invalid_pool_method():
    """Test that invalid pool_method raises ValueError."""
    with pytest.raises(ValueError, match="pool_method must be one of"):
        VanillaTransformer(in_features=1280, pool_method="invalid")


def test_vanilla_transformer_invalid_input_dim():
    """Test that inputs with wrong number of dimensions raise ValueError."""
    model = VanillaTransformer(in_features=1280)
    x_4d = torch.randn(2, 3, 50, 1280)
    with pytest.raises(ValueError):
        model(x_4d)


def test_vanilla_transformer_invalid_feature_dim():
    """Test that mismatched feature dimension raises ValueError."""
    model = VanillaTransformer(in_features=1280)
    x_wrong_dim = torch.randn(2, 50, 512)
    with pytest.raises(ValueError):
        model(x_wrong_dim)


def test_vanilla_transformer_dropout_param():
    """Test that dropout is applied in the model."""
    model = VanillaTransformer(in_features=64, hidden_dim=128, out_features=1, dropout=0.5)
    dropout_layers = [m for m in model.classifier.modules() if isinstance(m, torch.nn.Dropout)]
    assert len(dropout_layers) == 1
    assert dropout_layers[0].p == 0.5


def test_vanilla_transformer_different_architectures():
    """Test various architectural configurations."""
    x = torch.randn(2, 30, 512)

    # Small model
    model_small = VanillaTransformer(
        in_features=512, hidden_dim=1024, num_heads=4, num_layers=1, out_features=3
    )
    out_small = model_small(x)
    assert out_small["logits"].shape == (2, 3)

    # Larger model
    model_large = VanillaTransformer(
        in_features=512, hidden_dim=2048, num_heads=8, num_layers=6, out_features=3
    )
    out_large = model_large(x)
    assert out_large["logits"].shape == (2, 3)


def test_vanilla_transformer_gradient_flow():
    """Test that gradients flow through the model."""
    model = VanillaTransformer(in_features=256, out_features=1, num_layers=2)
    x = torch.randn(2, 20, 256, requires_grad=True)

    out = model(x)
    loss = out["logits"].sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape

    # Check that model parameters have gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_vanilla_transformer_eval_mode():
    """Test that model works in eval mode and disables dropout."""
    model = VanillaTransformer(in_features=256, out_features=2, dropout=0.5)
    model.eval()

    x = torch.randn(2, 25, 256)

    # Run twice in eval mode - should be deterministic
    out1 = model(x)
    out2 = model(x)

    assert torch.allclose(out1["logits"], out2["logits"])


def test_vanilla_transformer_large_sequence():
    """Test with a large sequence (simulating full WSI)."""
    model = VanillaTransformer(in_features=1280, num_layers=3)
    x = torch.randn(1, 500, 1280)  # Large slide with 500 patches

    out = model(x)

    assert out["logits"].shape == (1, 1)


def test_vanilla_transformer_single_patch():
    """Test with minimal sequence (single patch)."""
    model = VanillaTransformer(in_features=1280)
    x = torch.randn(2, 1, 1280)  # Batch of 2 slides, single patch each

    out = model(x)

    assert out["logits"].shape == (2, 1)


def test_vanilla_transformer_cls_token_learnable():
    """Test that CLS token is a learnable parameter."""
    model = VanillaTransformer(in_features=256, pool_method="cls")

    assert isinstance(model.cls_token, torch.nn.Parameter)
    assert model.cls_token.requires_grad


def test_vanilla_transformer_consistency_across_modes():
    """Test that model produces consistent output shapes across pool methods."""
    x = torch.randn(3, 60, 768)

    for pool_method in ["cls", "mean", "max"]:
        model = VanillaTransformer(
            in_features=768, out_features=4, pool_method=pool_method, num_layers=1
        )
        out = model(x)
        assert out["logits"].shape == (3, 4)
