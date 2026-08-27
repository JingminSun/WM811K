"""

Train the wafer-map defect classifier.

"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from model.eval import predict, summarize
from model.model import build_model, count_parameters
from model.read_data import build_loaders, class_weights, load_prepared, make_splits


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/LSWMD_64.npz", help="prepared npz from prepare_data.py")
    ap.add_argument("--outdir", default="outputs/",
                    help="output directory")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.005, help="peak LR")
    ap.add_argument("--weight-decay", type=float, default=5e-2)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of the training half held out for validation")
    ap.add_argument("--seed", type=int, default=1111)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return ap.parse_args()


def describe_splits(data, train_idx, val_idx, test_idx):
    y, lots, classes = data["y"], data["lots"], data["classes"]

    print(f"\ntrain {len(train_idx):,} | val {len(val_idx):,} | test {len(test_idx):,}\n")

    train_lots, val_lots = set(lots[train_idx]), set(lots[val_idx])

    print(f"\ntrain lots {len(train_lots):,} | val lots {len(val_lots):,}\n")

    print(f"\n{'class':<12}{'train':>9}{'val':>8}{'test':>9}")
    for i, cls in enumerate(classes):
        print(f"{cls:<12}{int((y[train_idx] == i).sum()):>9,}"
              f"{int((y[val_idx] == i).sum()):>8,}"
              f"{int((y[test_idx] == i).sum()):>9,}")

    missing = [classes[i] for i in range(len(classes)) if (y[val_idx] == i).sum() == 0]
    if missing:
        print(f"\nWarning: validation has no examples of {missing}; "
              f"raise --val-frac")


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss, n_seen = 0.0, 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        loss = criterion(model(x), y)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * len(y)
        n_seen += len(y)

    return total_loss / max(n_seen, 1)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)

    data = load_prepared(args.data)
    classes = data["classes"]
    n_classes = len(classes)

    train_idx, val_idx, test_idx = make_splits(data, val_frac=args.val_frac, seed=args.seed)
    describe_splits(data, train_idx, val_idx, test_idx)

    train_loader, val_loader, _ = build_loaders(
        data, train_idx, val_idx, test_idx,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    weights = class_weights(data["y"][train_idx], n_classes, scheme="sqrt")
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    model = build_model(n_classes=n_classes, dropout=args.dropout).to(device)
    print(f"\nmodel: {count_parameters(model):,} parameters on {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    steps_per_epoch =  len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=args.epochs * steps_per_epoch,
    )

    log_path = os.path.join(args.outdir, "train_log.jsonl")
    best_f1, best_epoch = -1.0, -1
    started = time.time()

    with open(log_path, "w") as log:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                   scheduler, device)

            y_true, y_pred = predict(model, val_loader, device)
            val = summarize(y_true, y_pred, n_classes)

            record = {
                "epoch": epoch,
                "train_loss": loss,
                "val_mean_f1": val["mean_f1"],
                "val_accuracy": val["accuracy"],
                "val_balanced_accuracy": val["balanced_accuracy"],
                "lr": scheduler.get_last_lr()[0],
                "seconds": time.time() - t0,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()

            marker = ""
            if val["mean_f1"] > best_f1:
                best_f1, best_epoch = val["mean_f1"], epoch
                torch.save({
                    "model": model.state_dict(),
                    "classes": classes,
                    "epoch": epoch,
                    "val_mean_f1": best_f1,
                    "args": vars(args),
                }, os.path.join(args.outdir, "best.pt"))
                marker = " * new best"

            print(f"epoch {epoch:>3}/{args.epochs}  loss {loss:.4f}  "
                  f"validation mean-F1 {val['mean_f1']:.4f}  acc {val['accuracy']:.4f}  "
                  f"bal-acc {val['balanced_accuracy']:.4f}  "
                  f"{record['seconds']:.1f}s{marker}")


    print(f"\nbest validation mean-F1 {best_f1:.4f} at epoch {best_epoch} "
          f"({(time.time() - started)} sec total)")

if __name__ == "__main__":
    main()
