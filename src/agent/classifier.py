"""

Inference wrapper around the trained checkpoint.

"""

import base64
import io
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from model.model import build_model
from utils.prepare_data import resize_for_training


_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CKPT = os.environ.get("WAFER_CKPT", os.path.join(_ROOT, "outputs", "best.pt"))

_model = None
_classes = None


def load_model(ckpt_path=None, device=None):
    global _model, _classes

    if _model is not None:
        return _model, _classes

    path = ckpt_path or _CKPT
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    _classes = ckpt["classes"]

    _model = build_model(n_classes=len(_classes)).to(device)
    _model.load_state_dict(ckpt["model"])
    _model.eval()
    _model._device = device

    return _model, _classes


def to_wafer_map(wafer_map):
    """
    The input should be one form of npy file.     

    Values are the raw die codes: 0 = off-wafer, 1 = passing, 2 = failing.
    """
    text = wafer_map.strip()
    if text.endswith(".npy"):
        wm = np.load(text)
    else:
        wm = np.load(io.BytesIO(base64.b64decode(text)), allow_pickle=False)
   
    wm = np.squeeze(wm)
    if wm.ndim != 2 or wm.size == 0:
        raise ValueError(f"expected a 2-D wafer map, got shape {wm.shape}")

    return wm.astype(np.uint8)


@torch.no_grad()
def classify(wafer_map, top_k=3):

    model, classes = load_model()

    wm = to_wafer_map(wafer_map)
    rendered = resize_for_training(wm, target_size=64)

    x = torch.from_numpy(rendered).unsqueeze(0).to(model._device)
    probs = F.softmax(model(x).float(), dim=1)[0].cpu().numpy()

    order = probs.argsort()[::-1]
    return {
        "predicted_class": classes[order[0]],
        "confidence": round(float(probs[order[0]]), 4),
        "top_k": [
            {"class": classes[i], "probability": round(float(probs[i]), 4)}
            for i in order[:top_k]
        ],
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
        "input_shape": list(wm.shape),
    }


if __name__ == "__main__":
    pass
