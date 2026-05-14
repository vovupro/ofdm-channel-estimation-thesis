"""
Tiện ích dùng chung cho toàn bộ pipeline benchmark.
"""
import os
import json
import random
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml


def set_seed(seed: int = 42) -> None:
    """Cố định random seed cho numpy, random, và TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def load_config(yaml_path: str) -> Dict[str, Any]:
    """Đọc file YAML config và trả về dict."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str, data: Any) -> None:
    """Lưu dict/list ra file JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Any:
    """Đọc file JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def nmse_db(h_true, h_hat) -> float:
    """
    Tính NMSE (dB): 10*log10(E[||H-H_hat||^2] / E[||H||^2]).
    Nhận numpy array hoặc TF tensor.
    """
    try:
        import tensorflow as tf
        h_true = tf.cast(tf.squeeze(h_true), tf.complex64)
        h_hat  = tf.cast(tf.squeeze(h_hat),  tf.complex64)
        num = tf.reduce_mean(tf.square(tf.abs(h_true - h_hat)))
        den = tf.reduce_mean(tf.square(tf.abs(h_true))) + 1e-12
        nmse = (num / den).numpy()
    except Exception:
        h_true = np.squeeze(np.asarray(h_true))
        h_hat  = np.squeeze(np.asarray(h_hat))
        num = np.mean(np.abs(h_true - h_hat) ** 2)
        den = np.mean(np.abs(h_true) ** 2) + 1e-12
        nmse = num / den
    return float(10.0 * np.log10(nmse + 1e-12))


def get_logger(name: str) -> logging.Logger:
    """Logger đơn giản ra stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def ensure_dirs(paths: List[str]) -> None:
    """Tạo các thư mục nếu chưa tồn tại."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def patch_sionna_llvm() -> bool:
    """
    Patch sionna/__init__.py để bỏ qua lỗi import rt (cần LLVM) trên Windows.
    Trả về True nếu đã patch thành công.
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec("sionna")
        if spec is None:
            return False
        init_path = os.path.join(os.path.dirname(spec.origin), "__init__.py")
        with open(init_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Chỉ patch nếu chưa có try-except cho import rt
        if "from . import rt" in content and "try:\n    from . import rt" not in content:
            content = content.replace(
                "from . import rt",
                "try:\n    from . import rt\nexcept Exception:\n    pass  # LLVM không có trên Windows",
            )
            with open(init_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        return False
    except Exception:
        return False
