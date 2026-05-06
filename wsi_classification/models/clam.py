import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GatedAttention(nn.Module):
    """Gated attention mechanism for MIL."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_b = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attention_c = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b)
        return A


class CLAM_SB(nn.Module):
    """CLAM Single-Branch: Clustering-constrained Attention MIL (Li et al., 2021).

    Single attention branch applied to all instances.

    Args:
        gate: Use gated attention mechanism
        size_arg: Model size 'small' or 'big'
        dropout: Dropout rate
        k_sample: Number of top instances to sample for instance-level loss
        n_classes: Number of output classes
        instance_loss_fn: Loss function for instance classification
        subtyping: Enable subtyping mode
        embed_dim: Input feature dimension
    """

    def __init__(
        self,
        gate: bool = True,
        size_arg: str = "small",
        dropout: float = 0.0,
        k_sample: int = 8,
        n_classes: int = 2,
        instance_loss_fn: Optional[nn.Module] = None,
        subtyping: bool = False,
        embed_dim: int = 1024,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        if in_features is not None:
            embed_dim = in_features
        self.size_arg = size_arg
        self.n_classes = n_classes
        self.out_features = n_classes
        self.instance_loss_fn = instance_loss_fn
        self.subtyping = subtyping
        self.embed_dim = embed_dim
        self.gate = gate
        self.k_sample = k_sample

        # Layer sizes based on small/big configuration
        if size_arg == "small":
            hidden = [embed_dim, 512, 256]
        elif size_arg == "big":
            hidden = [embed_dim, 512, 384]
        else:
            raise ValueError(f"Invalid size_arg: {size_arg}")

        # Feature projection
        fc = [nn.Linear(hidden[0], hidden[1]), nn.ReLU()]
        if dropout > 0:
            fc.append(nn.Dropout(dropout))
        self.feature_extractor = nn.Sequential(*fc)

        # Attention mechanism
        if gate:
            self.attention_net = GatedAttention(hidden[1], hidden[2], dropout)
        else:
            self.attention_net = nn.Sequential(
                nn.Linear(hidden[1], hidden[2]),
                nn.Tanh(),
                nn.Linear(hidden[2], 1),
            )

        # Instance classifiers for each class
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(hidden[1], 2) for _ in range(n_classes)]
        )

        # Bag classifier
        self.bag_classifiers = nn.ModuleList(
            [nn.Linear(hidden[1], 1) for _ in range(n_classes)]
        )

    def forward(
        self,
        x: torch.Tensor,
        instance_eval: bool = False,
        return_features: bool = False,
        attention_only: bool = False,
    ) -> dict:
        """Forward pass.

        Args:
            x: Tensor of shape (N, D) where N is number of instances
            instance_eval: Whether to compute instance-level predictions
            return_features: Whether to return aggregated features
            attention_only: Whether to return only attention weights

        Returns:
            Dictionary with keys: logits, probs, pred, attention, instance_loss, features
        """
        # Ensure 2D input
        if x.dim() == 3:
            x = x.squeeze(0)

        assert x.dim() == 2, f"Expected 2D input, got {x.dim()}D"

        # Feature extraction
        h = self.feature_extractor(x)  # (N, hidden[1])

        # Attention
        A = self.attention_net(h)  # (N, 1)
        A = torch.transpose(A, 1, 0)  # (1, N)
        A = F.softmax(A, dim=1)  # (1, N)

        if attention_only:
            return {"attention": A}

        # Aggregate features via attention
        M = torch.mm(A, h)  # (1, hidden[1])

        logits = torch.empty(1, self.n_classes).to(x.device)
        for i in range(self.n_classes):
            logits[0, i] = self.bag_classifiers[i](M).squeeze()

        # Keep batch dimension: shape (1, n_classes)

        # For binary classification (n_classes=1), convert to binary logits
        if self.n_classes == 1:
            logits = torch.cat([torch.zeros_like(logits), logits], dim=0)

        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)

        out = {
            "logits": logits,
            "probs": probs,
            "pred": pred,
            "attention": A,
        }

        if return_features:
            out["features"] = M

        # Instance-level predictions for clustering loss
        if instance_eval:
            instance_logits = torch.empty(x.shape[0], self.n_classes, 2).to(x.device)
            for i in range(self.n_classes):
                instance_logits[:, i, :] = self.instance_classifiers[i](h)
            out["instance_logits"] = instance_logits

        return out


class CLAM_MB(CLAM_SB):
    """CLAM Multi-Branch: Class-specific attention branches.

    Each class has its own attention mechanism and classifier.
    Inherits from CLAM_SB with modified forward pass.
    """

    def __init__(
        self,
        gate: bool = True,
        size_arg: str = "small",
        dropout: float = 0.0,
        k_sample: int = 8,
        n_classes: int = 2,
        instance_loss_fn: Optional[nn.Module] = None,
        subtyping: bool = False,
        embed_dim: int = 1024,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        if in_features is not None:
            embed_dim = in_features
        super().__init__(
            gate=gate,
            size_arg=size_arg,
            dropout=dropout,
            k_sample=k_sample,
            n_classes=n_classes,
            instance_loss_fn=instance_loss_fn,
            subtyping=subtyping,
            embed_dim=embed_dim,
        )

        # Class-specific attention networks
        if size_arg == "small":
            hidden = [embed_dim, 512, 256]
        else:
            hidden = [embed_dim, 512, 384]

        self.attention_nets = nn.ModuleList(
            [
                GatedAttention(hidden[1], hidden[2], dropout) if gate
                else nn.Sequential(
                    nn.Linear(hidden[1], hidden[2]),
                    nn.Tanh(),
                    nn.Linear(hidden[2], 1),
                )
                for _ in range(n_classes)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        instance_eval: bool = False,
        return_features: bool = False,
        attention_only: bool = False,
    ) -> dict:
        """Forward pass with class-specific attention."""
        if x.dim() == 3:
            x = x.squeeze(0)

        assert x.dim() == 2, f"Expected 2D input, got {x.dim()}D"

        # Feature extraction
        h = self.feature_extractor(x)  # (N, hidden[1])

        logits = torch.empty(self.n_classes).to(x.device)
        attention_weights = []
        aggregated_features = []

        # Class-specific attention and aggregation
        for i in range(self.n_classes):
            A = self.attention_nets[i](h)  # (N, 1)
            A = torch.transpose(A, 1, 0)  # (1, N)
            A = F.softmax(A, dim=1)  # (1, N)
            attention_weights.append(A)

            # Aggregate features
            M = torch.mm(A, h)  # (1, hidden[1])
            aggregated_features.append(M)
            logits[i] = self.bag_classifiers[i](M).squeeze()

        attention_weights = torch.cat(attention_weights, dim=0)  # (n_classes, N)

        if attention_only:
            return {"attention": attention_weights}

        # Reshape logits to (1, n_classes) to maintain batch dimension
        logits = logits.unsqueeze(0)  # (1, n_classes)

        # For binary classification, convert to 2-class logits
        if self.n_classes == 1:
            logits = torch.cat([torch.zeros_like(logits), logits], dim=1)

        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)

        out = {
            "logits": logits,
            "probs": probs,
            "pred": pred,
            "attention": attention_weights,
        }

        if return_features:
            # Return average of class-specific aggregations
            M_avg = torch.mean(torch.cat(aggregated_features, dim=0), dim=0)
            out["features"] = M_avg

        if instance_eval:
            instance_logits = torch.empty(x.shape[0], self.n_classes, 2).to(x.device)
            for i in range(self.n_classes):
                instance_logits[:, i, :] = self.instance_classifiers[i](h)
            out["instance_logits"] = instance_logits

        return out


# Legacy alias for backward compatibility
CLAM = CLAM_SB
