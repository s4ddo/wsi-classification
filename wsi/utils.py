import torch
import torch.nn.functional as F


def collate_wsi(batch, win_size=None):
    """Does not work yet."""
    raise NotImplementedError

    features = [item["input"] for item in batch]
    coords = [item["coords"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])

    grid_dims = []
    for c in coords:
        # Number of unique coordinates defines the grid size
        h = len(torch.unique(c[:, 1]))
        w = len(torch.unique(c[:, 0]))
        grid_dims.append((h, w))

    max_h = max(d[0] for d in grid_dims)
    max_w = max(d[1] for d in grid_dims)

    if win_size is not None:
        max_h = ((max_h + win_size - 1) // win_size) * win_size
        max_w = ((max_w + win_size - 1) // win_size) * win_size

    C = features[0].shape[1]
    batch_size = len(batch)

    padded_features = torch.zeros(batch_size, max_h * max_w, C)
    padded_coords = torch.zeros(batch_size, max_h * max_w, 2)
    attn_mask = torch.zeros(batch_size, max_h * max_w)

    for i, f in enumerate(features):
        n = f.shape[0]
        padded_features[i, :n, :] = f
        padded_coords[i, :n, :] = coords[i]
        attn_mask[i, :n] = 1

    return {
        "input": padded_features,
        "mask": attn_mask,
        "label": labels,
        "coords": padded_coords,
        "grid_size": (max_h, max_w)
    }


def apply_embedding_dropout(embeddings, dropout_rate=0.9):
    return F.dropout(embeddings, p=dropout_rate, training=True)
