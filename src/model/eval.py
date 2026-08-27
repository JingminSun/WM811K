"""

Evaluate a trained checkpoint on the held-out test split.

"""

import argparse
import json
import os

import numpy as np
import torch


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_prf1(cm):
    tp = np.diag(cm).astype(np.float32)
    predicted = cm.sum(axis=0).astype(np.float32)
    actual = cm.sum(axis=1).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(actual > 0, tp / actual, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)

    return precision, recall, f1, actual.astype(np.int64)


def summarize(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    precision, recall, f1, support = per_class_prf1(cm)

    present = support > 0

    return {
        "accuracy": float((y_true == y_pred).mean()),
        "correct": int(np.diag(cm).sum()),
        "n": int(cm.sum()),
        "mean_f1": float(f1[present].mean()),
        "balanced_accuracy": float(recall[present].mean()),
        "per_class": {
            "precision": precision,
            "recall": recall,
            "correct": np.diag(cm).astype(np.int64),
            "f1": f1,
            "support": support,
        },
        "confusion_matrix": cm,
    }


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, targets = [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds.append(logits.float().argmax(1).cpu().numpy())
        targets.append(y.numpy())

    return np.concatenate(targets), np.concatenate(preds)


def print_report(result, classes, title="test"):
    print(f"\n{title}")
    print(f"  overall accuracy   {result['accuracy']:.4f}"
          f"   ({result['correct']:,} / {result['n']:,} correct)")
    print(f"  balanced accuracy  {result['balanced_accuracy']:.4f}"
          f"   (mean of the per-class recalls below)")
    print(f"  mean-F1            {result['mean_f1']:.4f}")

    pc = result["per_class"]
    print(f"\nper-class recall (correct / support for that class):")
    print(f"\n{'class':<12}{'recall':>10}{'correct':>10}{'support':>10}"
          f"{'prec':>8}{'f1':>8}")
    for i, cls in enumerate(classes):
        print(f"{cls:<12}{pc['recall'][i]:>10.4f}{pc['correct'][i]:>10,}"
              f"{pc['support'][i]:>10,}{pc['precision'][i]:>8.4f}{pc['f1'][i]:>8.4f}")

    print("\nconfusion matrix, counts (rows = true, cols = predicted):")
    cm = result["confusion_matrix"]
    header = "".join(f"{c[:7]:>9}" for c in classes)
    print(f"{'':<12}{header}")
    for i, cls in enumerate(classes):
        print(f"{cls:<12}" + "".join(f"{v:>9,}" for v in cm[i]))

    print("\nconfusion matrix, row-normalised (each row sums to 1;"
          " the diagonal is the per-class recall):")
    print(f"{'':<12}{header}")
    for i, cls in enumerate(classes):
        row = cm[i] / cm[i].sum() if cm[i].sum() else cm[i]
        print(f"{cls:<12}" + "".join(f"{v:>9.3f}" for v in row))


def save_confusion_png(cm, classes, path, normalize=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = cm.astype(np.float32)
    if normalize:
        row_sums = m.sum(axis=1, keepdims=True)
        m = np.divide(m, row_sums, out=np.zeros_like(m), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    if normalize:
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
        label = "row-normalised share (diagonal = per-class recall)"
    else:
        from matplotlib.colors import LogNorm
        im = ax.imshow(np.maximum(m, 0.5), cmap="Blues", norm=LogNorm(vmin=0.5, vmax=max(m.max(), 1)))
        label = "count (log scale)"
    fig.colorbar(im, ax=ax, label=label)

    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix" + ("row-normalised: recall" if normalize else " (counts)"))

    threshold = 0.5 if normalize else m.max() ** 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{m[i, j]:.2f}" if normalize else f"{cm[i, j]:,}",
                    ha="center", va="center",
                    color="white" if m[i, j] > threshold else "black", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    from model.model import build_model
    from model.read_data import build_loaders, load_prepared, make_splits

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/LSWMD_64.npz", help="prepared npz from prepare_data.py")
    ap.add_argument("--ckpt", default="outputs/best.pt", help="checkpoint to evaluate")
    ap.add_argument("--outdir", default=None, help="where to write the report (default: the checkpoint's directory)")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    trained_with = ckpt.get("args", {})

    outdir = args.outdir or os.path.dirname(args.ckpt) or "."
    os.makedirs(outdir, exist_ok=True)

    data = load_prepared(args.data)

    train_idx, val_idx, test_idx = make_splits(
        data,
        val_frac=trained_with.get("val_frac", 0.15),
        seed=trained_with.get("seed", 0),
    )
    _, _, test_loader = build_loaders(
        data, train_idx, val_idx, test_idx,
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = build_model(n_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model"])

    y_true, y_pred = predict(model, test_loader, device)

    result = summarize(y_true, y_pred, len(classes))
    print(f"checkpoint: {args.ckpt}, "
          f"validation mean-F1 {ckpt.get('val_mean_f1', float('nan')):.4f})")
    print_report(result, classes, title=f"[{len(y_true):,} wafers]")

    report_path = os.path.join(outdir, f"report_eval.json")
    with open(report_path, "w") as f:
        json.dump({
            "checkpoint": args.ckpt,
            "n": int(len(y_true)),
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "mean_f1": result["mean_f1"],
            "balanced_accuracy": result["balanced_accuracy"],
            "classes": classes,
            "per_class_correct": result["per_class"]["correct"].tolist(),
            "precision": result["per_class"]["precision"].tolist(),
            "recall": result["per_class"]["recall"].tolist(),
            "f1": result["per_class"]["f1"].tolist(),
            "support": result["per_class"]["support"].tolist(),
            "confusion_matrix": result["confusion_matrix"].tolist(),
        }, f, indent=2)

    cm_path = os.path.join(outdir, "confusion_eval.png")
    save_confusion_png(result["confusion_matrix"], classes, cm_path)

    cm_counts_path = os.path.join(outdir, "confusion_eval_counts.png")
    save_confusion_png(result["confusion_matrix"], classes, cm_counts_path,
                       normalize=False)



if __name__ == "__main__":
    main()
