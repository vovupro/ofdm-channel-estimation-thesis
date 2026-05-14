"""Tiện ích dùng chung cho pipeline benchmark."""
import os, json, random
import numpy as np
import yaml


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def load_config(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def nmse_db(h_true, h_hat) -> float:
    """NMSE (dB) = 10·log10(E[‖H−Ĥ‖²] / E[‖H‖²])."""
    h_true = np.squeeze(np.asarray(h_true))
    h_hat  = np.squeeze(np.asarray(h_hat))
    num = np.mean(np.abs(h_true - h_hat) ** 2)
    den = np.mean(np.abs(h_true) ** 2) + 1e-12
    return float(10.0 * np.log10(num / den + 1e-12))
