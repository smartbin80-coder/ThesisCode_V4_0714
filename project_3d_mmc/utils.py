"""通用工具函数。"""

import json
import random
import time
from functools import wraps
from pathlib import Path

import numpy as np


def ensure_dir(path):
    """确保目录存在。"""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, data):
    """保存 JSON 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path):
    """读取 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_features(x, eps=1e-12):
    """按列标准化特征矩阵。"""
    x = np.asarray(x, dtype=float)
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + eps)


def set_random_seed(seed):
    """固定随机种子，便于复现实验。"""
    random.seed(seed)
    np.random.seed(seed)


def timer(func):
    """打印函数运行时间的装饰器。"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        out = func(*args, **kwargs)
        print(f"{func.__name__} 用时 {time.time() - start:.2f} s")
        return out
    return wrapper
