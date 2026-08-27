"""

This file is to pre-process the data for the model training and evaluation, 
based on the data analyzis in the notebook "data_analysis.ipynb". 

"""

import multiprocessing
import os
import argparse

import cv2
import numpy as np
import pandas as pd

import pickle

CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
            "Near-full", "Random", "Scratch", "none"
        ]

def normalize_label(value):
    if isinstance(value,str):
        return value.strip() or "unlabeled"
    arr = np.asarray(value)
    if arr.size == 0:
        return "unlabeled"
    return str(arr.ravel()[0]).strip() or "unlabeled"


def resize_for_training(wafer_map, target_size=64):
    wm = np.asarray(wafer_map)
    if wm.ndim != 2 or wm.size == 0:
        return None

    out = np.empty((2, target_size, target_size), dtype=np.float32)

    shrinking = wm.shape[0] > target_size and wm.shape[1] > target_size
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR

    for i, dieval in enumerate((1,2)):
        mask = (wm == dieval).astype(np.float32)
        out[i] = cv2.resize(mask, (target_size, target_size), interpolation=interpolation)

    return out

class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("pandas.indexes"):
            module = "pandas.core." + module[len("pandas."):]
        return super().find_class(module, name)


SIZE = None
def _init(size):
    global SIZE
    SIZE = size

def _work(wafer_map):
    return resize_for_training(wafer_map, SIZE)


def main():

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pkl", default="data/LSWMD.pkl", help="path to the input pickle file")
    ap.add_argument("--zip", default="data/LSWMD.pkl.zip", help="path to the input zip file")
    ap.add_argument("--kaggle-ds", default="qingyi/wm811k-wafer-map", help="Kaggle dataset identifier for downloading the data")
    ap.add_argument("--output", default="data/LSWMD_64.npz", help="path to the output npz file")
    ap.add_argument("--size", type=int, default=64, help="target size for resizing the wafer maps")
    ap.add_argument("--num-workers", type=int, default=min(16, os.cpu_count()), help="number of workers for parallel processing")

    args = ap.parse_args()

    if os.path.exists(args.pkl):
        print(f"Found {args.pkl}, skipping download.")
    else:
        import zipfile
        os.system(f"kaggle datasets download -d {args.kaggle_ds} -f LSWMD.pkl -p data")
        with zipfile.ZipFile(args.zip) as z:
            z.extractall("data")
        os.remove(args.zip)





    with open(args.pkl, "rb") as f:
        df = _LegacyUnpickler(f, encoding="latin1").load()

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected a pandas DataFrame, but got {type(df)}")

    print(f"Loaded DataFrame with shape: {df.shape}")

    labels = df["failureType"].map(normalize_label)
    keep = labels.isin(CLASSES)

    dropped_unlabeled = int((labels == "unlabeled").sum())
    dropped_other = int(((labels != "unlabeled") & ~keep).sum())

    print(f"Dropping {dropped_unlabeled} unlabeled samples and {dropped_other} samples with other labels.")

    df_sub = df.loc[keep]
    labels = labels.loc[keep]
    print(f"Remaining samples after filtering: {len(df_sub)}")

    print(f"Resizing wafer maps to {args.size}x{args.size}...")

    wafer_maps = list(df_sub["waferMap"])
    X = np.empty((len(wafer_maps), 2, args.size, args.size), dtype=np.float32)
    kept_rows = []

    with multiprocessing.Pool(processes=args.num_workers, initializer=_init, initargs=(args.size,)) as pool:
        for i, rendered in enumerate(pool.imap(_work, wafer_maps, chunksize=256)):
            if rendered is None:
                continue
            X[len(kept_rows)] = rendered
            kept_rows.append(i)

    if len(kept_rows) < len(wafer_maps):
        print(f"Warning: {len(wafer_maps) - len(kept_rows)} wafer maps could not be resized and will be skipped.")

    X = X[:len(kept_rows)]
    class_to_index = {cls: idx for idx, cls in enumerate(CLASSES)}
    y = np.array([class_to_index[labels.iloc[i]] for i in kept_rows], dtype=np.int64)

 
    lots = np.asarray(df_sub["lotName"].astype(str).to_numpy()[kept_rows], dtype=np.str_)


    split = np.asarray(df_sub["trianTestLabel"].map(normalize_label).to_numpy()[kept_rows], dtype=np.str_)
    is_train = split == "Training"
    is_test = split == "Test"

    print(f"\nFinal dataset shape: X={X.shape} ({X.dtype}), y={y.shape}, lots={lots.shape}")

    print(f"\nSplit from trianTestLabel: {is_train.sum():,} train / {is_test.sum():,} test")
    if not (is_train | is_test).all():
        print(f"Warning: {(~(is_train | is_test)).sum()} kept samples carry no train/test flag.")

    shared_lots = set(lots[is_train]) & set(lots[is_test])
    print(f"Lots: {len(set(lots[is_train])):,} train / {len(set(lots[is_test])):,} test / {len(shared_lots)} shared")

    print(f"\n{'class':<12}{'train':>9}{'test':>9}")
    for cls, idx in class_to_index.items():
        print(f"{cls:<12}{int(((y == idx) & is_train).sum()):>9,}{int(((y == idx) & is_test).sum()):>9,}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    print(f"\nWriting {args.output} ...")
    np.savez_compressed(
        args.output,
        X=X,
        y=y,
        lots=lots,
        split=split,
        classes=np.array(CLASSES),
        size=np.int64(args.size),
    )

if __name__ == "__main__":
    main()

