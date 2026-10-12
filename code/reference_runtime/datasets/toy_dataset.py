import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------------------------------
# dataset
# --------------------------------------------------
class Toy2DDataset(Dataset):
    def __init__(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(f"x must have shape [N, 2], got {x.shape}")
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]


# --------------------------------------------------
# config
# --------------------------------------------------
@dataclass
class ShiftConfig:
    rotation_deg: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    shear: float = 0.0
    translate_x: float = 0.0
    translate_y: float = 0.0


# --------------------------------------------------
# helpers
# --------------------------------------------------
def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_affine_matrix(cfg: ShiftConfig) -> np.ndarray:
    theta = math.radians(cfg.rotation_deg)
    rot = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)],
    ], dtype=np.float32)

    scale = np.array([
        [cfg.scale_x, 0.0],
        [0.0, cfg.scale_y],
    ], dtype=np.float32)

    shear = np.array([
        [1.0, cfg.shear],
        [0.0, 1.0],
    ], dtype=np.float32)

    return rot @ shear @ scale


def apply_affine_shift(x: np.ndarray, cfg: ShiftConfig) -> np.ndarray:
    A = _make_affine_matrix(cfg)
    t = np.array([cfg.translate_x, cfg.translate_y], dtype=np.float32)
    return x @ A.T + t


def _default_class_centers(
    num_classes: int,
    radius: float = 4.0,
    start_angle_deg: float = 0.0,
) -> np.ndarray:
    """
    클래스 중심을 원 위에 균등 배치.
    """
    angles = np.linspace(
        math.radians(start_angle_deg),
        math.radians(start_angle_deg) + 2 * math.pi,
        num_classes,
        endpoint=False,
    )
    centers = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles)],
        axis=1,
    )
    return centers.astype(np.float32)


# --------------------------------------------------
# main generator
# --------------------------------------------------
def make_gaussian_mixture_2d(
    n_samples: int,
    num_classes: int = 2,
    class_std: float = 0.8,
    radius: float = 4.0,
    class_probs: Optional[np.ndarray] = None,
    centers: Optional[np.ndarray] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D Gaussian mixture 생성.

    Returns:
        x: [N, 2]
        y: [N]
    """
    rng = np.random.default_rng(seed)

    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")

    if centers is None:
        centers = _default_class_centers(num_classes=num_classes, radius=radius)
    else:
        centers = np.asarray(centers, dtype=np.float32)
        if centers.shape != (num_classes, 2):
            raise ValueError(f"centers must have shape [{num_classes}, 2], got {centers.shape}")

    if class_probs is None:
        class_probs = np.ones(num_classes, dtype=np.float64) / num_classes
    else:
        class_probs = np.asarray(class_probs, dtype=np.float64)
        class_probs = class_probs / class_probs.sum()

    y = rng.choice(num_classes, size=n_samples, p=class_probs)
    x = np.zeros((n_samples, 2), dtype=np.float32)

    for c in range(num_classes):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        noise = rng.normal(loc=0.0, scale=class_std, size=(len(idx), 2)).astype(np.float32)
        x[idx] = centers[c] + noise

    return x, y.astype(np.int64)


def make_source_target_splits(
    n_source_train: int = 2000,
    n_source_test: int = 1000,
    n_target_test: int = 1000,
    num_classes: int = 2,
    class_std: float = 0.8,
    radius: float = 4.0,
    source_seed: int = 0,
    target_seed: int = 1,
    target_shift: Optional[ShiftConfig] = None,
) -> Dict[str, np.ndarray]:
    """
    source / target split 생성.
    target은 source와 같은 class-conditional structure를 가지되,
    전체 좌표계에 affine shift가 적용됨.
    """
    if target_shift is None:
        target_shift = ShiftConfig(
            rotation_deg=25.0,
            scale_x=1.15,
            scale_y=0.85,
            shear=0.20,
            translate_x=1.0,
            translate_y=-0.5,
        )

    centers = _default_class_centers(num_classes=num_classes, radius=radius)

    x_src_tr, y_src_tr = make_gaussian_mixture_2d(
        n_samples=n_source_train,
        num_classes=num_classes,
        class_std=class_std,
        centers=centers,
        seed=source_seed,
    )
    x_src_te, y_src_te = make_gaussian_mixture_2d(
        n_samples=n_source_test,
        num_classes=num_classes,
        class_std=class_std,
        centers=centers,
        seed=source_seed + 11,
    )
    x_tgt_te, y_tgt_te = make_gaussian_mixture_2d(
        n_samples=n_target_test,
        num_classes=num_classes,
        class_std=class_std,
        centers=centers,
        seed=target_seed,
    )

    x_tgt_te = apply_affine_shift(x_tgt_te, target_shift)

    return {
        "x_source_train": x_src_tr,
        "y_source_train": y_src_tr,
        "x_source_test": x_src_te,
        "y_source_test": y_src_te,
        "x_target_test": x_tgt_te,
        "y_target_test": y_tgt_te,
    }


# --------------------------------------------------
# torch dataset wrapper
# --------------------------------------------------
def build_toy_datasets(
    n_source_train: int = 2000,
    n_source_test: int = 1000,
    n_target_test: int = 1000,
    num_classes: int = 2,
    class_std: float = 0.8,
    radius: float = 4.0,
    source_seed: int = 0,
    target_seed: int = 1,
    target_shift: Optional[ShiftConfig] = None,
):
    splits = make_source_target_splits(
        n_source_train=n_source_train,
        n_source_test=n_source_test,
        n_target_test=n_target_test,
        num_classes=num_classes,
        class_std=class_std,
        radius=radius,
        source_seed=source_seed,
        target_seed=target_seed,
        target_shift=target_shift,
    )

    ds_source_train = Toy2DDataset(splits["x_source_train"], splits["y_source_train"])
    ds_source_test = Toy2DDataset(splits["x_source_test"], splits["y_source_test"])
    ds_target_test = Toy2DDataset(splits["x_target_test"], splits["y_target_test"])

    return ds_source_train, ds_source_test, ds_target_test, splits


# --------------------------------------------------
# example
# --------------------------------------------------
if __name__ == "__main__":
    set_seed(0)

    shift_cfg = ShiftConfig(
        rotation_deg=30.0,
        scale_x=1.2,
        scale_y=0.9,
        shear=0.15,
        translate_x=1.5,
        translate_y=-0.8,
    )

    ds_source_train, ds_source_test, ds_target_test, splits = build_toy_datasets(
        n_source_train=3000,
        n_source_test=1000,
        n_target_test=1000,
        num_classes=3,      # 여기 조절
        class_std=0.7,
        radius=4.5,
        target_shift=shift_cfg,
    )

    print("source train:", splits["x_source_train"].shape, splits["y_source_train"].shape)
    print("source test :", splits["x_source_test"].shape, splits["y_source_test"].shape)
    print("target test :", splits["x_target_test"].shape, splits["y_target_test"].shape)