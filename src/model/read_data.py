"""
Dataset loading for the prepared LSWMD_64.npz produced by prepare_data.py.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def load_prepared(path):
    d = np.load(path)
    return {
        "X": d["X"],
        "y": d["y"],
        "lots": d["lots"],
        "split": d["split"],
        "classes": [str(c) for c in d["classes"]],
        "size": int(d["size"]),
    }


def lot_disjoint_val_split(y, lots, n_classes, val_frac=0.15, seed=0):

    rng = np.random.default_rng(seed)
    lot_ids, lot_index = np.unique(lots, return_inverse=True)

    is_val_lot = np.zeros(len(lot_ids), dtype=bool)
    decided = np.zeros(len(lot_ids), dtype=bool)

    counts = np.bincount(y, minlength=n_classes)
    for cls in np.argsort(counts):
        if counts[cls] == 0:
            continue
        target = val_frac * counts[cls]

        rows_of_cls = y == cls
        in_val = int(rows_of_cls[is_val_lot[lot_index]].sum())

        candidate_lots = np.unique(lot_index[rows_of_cls])
        candidate_lots = candidate_lots[~decided[candidate_lots]]
        rng.shuffle(candidate_lots)

        for lot in candidate_lots:
            if in_val >= target:
                break
            is_val_lot[lot] = True
            decided[lot] = True
            in_val += int((rows_of_cls & (lot_index == lot)).sum())

        decided[candidate_lots] = True

    return is_val_lot[lot_index]


def make_splits(data, val_frac=0.15, seed=0): # One more split for validation during training

    y, lots, split = data["y"], data["lots"], data["split"]
    n_classes = len(data["classes"])

    train_rows = np.flatnonzero(split == "Training")
    test_rows = np.flatnonzero(split == "Test")

    val_mask = lot_disjoint_val_split(
        y[train_rows], lots[train_rows], n_classes, val_frac=val_frac, seed=seed
    )

    return train_rows[~val_mask], train_rows[val_mask], test_rows


class WaferDataset(Dataset):


    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        x = x.astype(np.float32)

        return torch.from_numpy(x), int(self.y[i])


def class_weights(y, n_classes, scheme="sqrt"):
 
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)

    if scheme == "none":
        w = np.ones(n_classes)
    elif scheme == "sqrt":
        w = np.sqrt(counts.sum() / counts)
    else:
        raise ValueError(f"unknown weighting scheme: {scheme}")

    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def make_sampler(y, n_classes):

    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = (1.0 / counts)[y]
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.float32),
        num_samples=len(y),
        replacement=True,
    )


def build_loaders(data, train_idx, val_idx, test_idx, batch_size=256,
                  num_workers=8):

    X, y = data["X"], data["y"]
    n_classes = len(data["classes"])

    train_ds = WaferDataset(X[train_idx], y[train_idx])
    val_ds = WaferDataset(X[val_idx], y[val_idx])
    test_ds = WaferDataset(X[test_idx], y[test_idx])


    common = dict(num_workers=num_workers, pin_memory=True,
                  persistent_workers=num_workers > 0)

    sampler = make_sampler(y[train_idx], n_classes)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)

    return train_loader, val_loader, test_loader
