"""

Pull wafer maps out of the LSWMD pickle and save them as .npy files.


    PYTHONPATH=src python3 -m utils.make_samples
    PYTHONPATH=src python3 -m utils.make_samples -n 50

"""

import argparse
import os
import pickle

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "samples")
PKL = os.path.join(ROOT, "data", "LSWMD.pkl")

UNLABELLED = "unlabelled"


class _LegacyUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if module.startswith("pandas.indexes"):
            module = "pandas.core." + module[len("pandas."):]
        return super().find_class(module, name)


def load_df(path):
    with open(path, "rb") as f:
        return _LegacyUnpickler(f, encoding="latin1").load()


def first_value(cell):
    """LSWMD stores labels as (1, 1) arrays, empty when absent."""
    arr = np.asarray(cell)
    return str(arr.ravel()[0]).strip() if arr.size else ""


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--count", type=int, default=20,
                    help="how many wafers to save (default 20)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    return ap.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    os.makedirs(OUT, exist_ok=True)

    for f in os.listdir(OUT):
        if f.endswith(".npy"):
            os.remove(os.path.join(OUT, f))


    df = load_df(PKL)
    labels = df["failureType"].map(first_value)
    splits = df["trianTestLabel"].map(first_value)
    print(f"{len(df):,} rows")

    is_train = (splits == "Training").to_numpy()
    is_test = (splits == "Test").to_numpy()
    is_unlabelled = (labels == "").to_numpy()

    print(f"  training  {is_train.sum():>8,}  (never sampled)")
    print(f"  test      {is_test.sum():>8,}")
    print(f"  unlabelled{is_unlabelled.sum():>8,}")

    pool = np.flatnonzero(is_test | is_unlabelled)
    take = min(args.count, len(pool))
    picks = rng.choice(pool, size=take, replace=False).tolist()
    how = f"{take} random from test + unlabelled"

    saved, counts = 0, {}
    for idx in picks:
        wm = np.asarray(df["waferMap"].iloc[idx])
        if wm.ndim != 2 or wm.size == 0:
            print(f"  skipping row {idx}: shape {wm.shape}")
            continue

        assert not is_train[idx], f"row {idx} is from the training split"

        wm = wm.astype(np.uint8)
        label = labels.iloc[idx] or UNLABELLED
        saved += 1
        np.save(os.path.join(OUT, f"sample{saved}.npy"), wm)
        counts[label] = counts.get(label, 0) + 1

    print(f"\nsaved {saved} wafers ({how}) to {os.path.relpath(OUT, ROOT)}/")
    for label in sorted(counts, key=lambda k: -counts[k]):
        print(f"  {label:<12}{counts[label]:>4}")


if __name__ == "__main__":
    main()
