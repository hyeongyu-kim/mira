import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances, silhouette_score
from scipy.stats import spearmanr


try:
    import umap
except Exception:
    umap = None


def unwrap_tta_model(model):
    return model.module if hasattr(model, "module") else model


def unwrap_base_model(model):
    m = unwrap_tta_model(model)
    return getattr(m, "model", m)


def find_last_linear_module(model):
    """
    Find the final nn.Linear module and use its input as the penultimate feature.

    This works for common torchvision/timm models:
    - ResNet: fc input
    - ViT: heads/head input
    - EfficientNet/ConvNeXt: classifier final Linear input
    - VGG: final classifier Linear input
    """
    base = unwrap_base_model(model)

    last_name = None
    last_module = None

    for name, module in base.named_modules():
        if isinstance(module, nn.Linear):
            last_name = name
            last_module = module

    if last_module is None:
        raise RuntimeError(
            "Could not find an nn.Linear classifier layer. "
            "Add a custom feature hook for this architecture."
        )

    return last_name, last_module


def _move_to_device(x, device):
    if isinstance(x, (list, tuple)):
        return [xx.to(device, non_blocking=True) if torch.is_tensor(xx) else xx for xx in x]
    return x.to(device, non_blocking=True)


def _prepare_labels(labels):
    if labels.ndim > 1:
        labels = labels.argmax(dim=1)
    return labels.long()


@torch.no_grad()
def collect_penultimate_features(
    model,
    data_loader,
    device,
    max_samples=-1,
    force_no_adapt=True,
    force_eval=True,
):
    """
    Collect features from the input to the last Linear layer.

    If model is a TTAMethod wrapper and force_no_adapt=True,
    it disables adaptation during this feature pass.
    """
    tta_model = unwrap_tta_model(model)
    base_model = unwrap_base_model(model)

    old_adapt_enabled = getattr(tta_model, "adaptation_enabled", None)
    old_training = base_model.training

    if force_no_adapt and hasattr(tta_model, "set_adaptation_enabled"):
        tta_model.set_adaptation_enabled(False)

    if force_eval:
        base_model.eval()

    layer_name, last_linear = find_last_linear_module(model)

    cache = {"feature": None}

    def pre_hook(module, inputs):
        feat = inputs[0]
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        if feat.ndim > 2:
            feat = torch.flatten(feat, 1)
        cache["feature"] = feat.detach().float().cpu()

    handle = last_linear.register_forward_pre_hook(pre_hook)

    features = []
    logits_list = []
    labels_list = []
    preds_list = []
    confs_list = []

    n_seen = 0

    try:
        for data in data_loader:
            imgs, labels = data[0], data[1]

            x_dev = _move_to_device(imgs, device)
            labels_cpu = _prepare_labels(labels).cpu()

            cache["feature"] = None
            logits = model(x_dev)

            if cache["feature"] is None:
                raise RuntimeError(
                    f"Feature hook did not fire. Last Linear layer was: {layer_name}"
                )

            probs = torch.softmax(logits.detach(), dim=1)
            confs, preds = probs.max(dim=1)

            feat = cache["feature"]
            logits_cpu = logits.detach().float().cpu()
            preds_cpu = preds.detach().cpu()
            confs_cpu = confs.detach().float().cpu()

            if max_samples is not None and max_samples > 0:
                remaining = max_samples - n_seen
                if remaining <= 0:
                    break

                feat = feat[:remaining]
                logits_cpu = logits_cpu[:remaining]
                labels_cpu = labels_cpu[:remaining]
                preds_cpu = preds_cpu[:remaining]
                confs_cpu = confs_cpu[:remaining]

            features.append(feat)
            logits_list.append(logits_cpu)
            labels_list.append(labels_cpu)
            preds_list.append(preds_cpu)
            confs_list.append(confs_cpu)

            n_seen += feat.shape[0]

            if max_samples is not None and max_samples > 0 and n_seen >= max_samples:
                break

    finally:
        handle.remove()

        if force_no_adapt and hasattr(tta_model, "set_adaptation_enabled") and old_adapt_enabled is not None:
            tta_model.set_adaptation_enabled(old_adapt_enabled)

        if force_eval and old_training:
            base_model.train()

    out = {
        "features": torch.cat(features, dim=0).numpy(),
        "logits": torch.cat(logits_list, dim=0).numpy(),
        "labels": torch.cat(labels_list, dim=0).numpy(),
        "preds": torch.cat(preds_list, dim=0).numpy(),
        "confs": torch.cat(confs_list, dim=0).numpy(),
        "feature_layer": layer_name,
    }

    return out


def l2_normalize_np(x, eps=1e-12):
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, eps)


def class_centroids(features, labels):
    classes = np.unique(labels)
    centroids = []
    valid_classes = []

    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        centroids.append(features[idx].mean(axis=0))
        valid_classes.append(c)

    return np.asarray(valid_classes), np.vstack(centroids)


def within_between_ratio(features, labels):
    classes, centroids = class_centroids(features, labels)
    if len(classes) < 2:
        return {
            "within_mean": np.nan,
            "between_mean": np.nan,
            "within_between_ratio": np.nan,
        }

    class_to_centroid = {int(c): centroids[i] for i, c in enumerate(classes)}

    within = []
    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        mu = class_to_centroid[int(c)][None, :]
        d = pairwise_distances(features[idx], mu, metric="cosine").reshape(-1)
        within.extend(d.tolist())

    centroid_d = pairwise_distances(centroids, metric="cosine")
    tri = np.triu_indices_from(centroid_d, k=1)
    between = centroid_d[tri]

    within_mean = float(np.mean(within)) if len(within) else np.nan
    between_mean = float(np.mean(between)) if len(between) else np.nan

    return {
        "within_mean": within_mean,
        "between_mean": between_mean,
        "within_between_ratio": float(within_mean / between_mean) if between_mean > 0 else np.nan,
    }


def compute_geometry_metrics(
    features_a,
    features_b,
    labels,
    max_pairwise_samples=2000,
    random_state=0,
):
    """
    Compare high-dimensional feature geometry.

    A: no adaptation
    B: adapted final
    """
    n = min(len(features_a), len(features_b), len(labels))
    features_a = features_a[:n]
    features_b = features_b[:n]
    labels = labels[:n]

    fa = l2_normalize_np(features_a.astype(np.float32))
    fb = l2_normalize_np(features_b.astype(np.float32))

    metrics = {}

    sample_cos = np.sum(fa * fb, axis=1)
    metrics["same_sample_cosine_mean"] = float(np.mean(sample_cos))
    metrics["same_sample_cosine_std"] = float(np.std(sample_cos))

    classes_a, cent_a = class_centroids(fa, labels)
    classes_b, cent_b = class_centroids(fb, labels)

    common = np.intersect1d(classes_a, classes_b)
    cent_a_common = np.vstack([cent_a[np.where(classes_a == c)[0][0]] for c in common])
    cent_b_common = np.vstack([cent_b[np.where(classes_b == c)[0][0]] for c in common])

    if len(common) >= 3:
        da = pairwise_distances(cent_a_common, metric="cosine")
        db = pairwise_distances(cent_b_common, metric="cosine")
        tri = np.triu_indices_from(da, k=1)

        rho = spearmanr(da[tri], db[tri]).correlation
        metrics["centroid_distance_spearman"] = float(rho)
        metrics["centroid_distance_abs_delta_mean"] = float(np.mean(np.abs(da[tri] - db[tri])))
    else:
        metrics["centroid_distance_spearman"] = np.nan
        metrics["centroid_distance_abs_delta_mean"] = np.nan

    centroid_shift = 1.0 - np.sum(
        l2_normalize_np(cent_a_common) * l2_normalize_np(cent_b_common),
        axis=1,
    )
    metrics["centroid_shift_cosine_mean"] = float(np.mean(centroid_shift))
    metrics["centroid_shift_cosine_max"] = float(np.max(centroid_shift))

    rng = np.random.default_rng(random_state)
    m = min(n, max_pairwise_samples)
    idx = rng.choice(n, size=m, replace=False)

    if m >= 10:
        pa = pairwise_distances(fa[idx], metric="cosine")
        pb = pairwise_distances(fb[idx], metric="cosine")
        tri = np.triu_indices_from(pa, k=1)
        rho = spearmanr(pa[tri], pb[tri]).correlation
        metrics["sample_pairwise_distance_spearman"] = float(rho)
        metrics["sample_pairwise_distance_abs_delta_mean"] = float(np.mean(np.abs(pa[tri] - pb[tri])))
    else:
        metrics["sample_pairwise_distance_spearman"] = np.nan
        metrics["sample_pairwise_distance_abs_delta_mean"] = np.nan

    wb_a = within_between_ratio(fa, labels)
    wb_b = within_between_ratio(fb, labels)

    for k, v in wb_a.items():
        metrics[f"noadapt_{k}"] = v
    for k, v in wb_b.items():
        metrics[f"adapted_{k}"] = v

    # Optional: silhouette on a balanced-ish subset.
    try:
        sil_m = min(n, 5000)
        sil_idx = rng.choice(n, size=sil_m, replace=False)
        if len(np.unique(labels[sil_idx])) >= 2:
            metrics["noadapt_silhouette_cosine"] = float(
                silhouette_score(fa[sil_idx], labels[sil_idx], metric="cosine")
            )
            metrics["adapted_silhouette_cosine"] = float(
                silhouette_score(fb[sil_idx], labels[sil_idx], metric="cosine")
            )
        else:
            metrics["noadapt_silhouette_cosine"] = np.nan
            metrics["adapted_silhouette_cosine"] = np.nan
    except Exception:
        metrics["noadapt_silhouette_cosine"] = np.nan
        metrics["adapted_silhouette_cosine"] = np.nan

    per_class = {}
    for i, c in enumerate(common):
        per_class[int(c)] = {
            "centroid_shift_cosine": float(centroid_shift[i]),
        }

    return metrics, per_class


def choose_plot_classes_by_shift(features_a, features_b, labels, max_classes=20):
    fa = l2_normalize_np(features_a.astype(np.float32))
    fb = l2_normalize_np(features_b.astype(np.float32))

    classes_a, cent_a = class_centroids(fa, labels)
    classes_b, cent_b = class_centroids(fb, labels)
    common = np.intersect1d(classes_a, classes_b)

    shifts = []
    for c in common:
        ca = cent_a[np.where(classes_a == c)[0][0]]
        cb = cent_b[np.where(classes_b == c)[0][0]]
        shift = 1.0 - np.dot(ca / max(np.linalg.norm(ca), 1e-12),
                             cb / max(np.linalg.norm(cb), 1e-12))
        shifts.append((int(c), float(shift)))

    shifts = sorted(shifts, key=lambda x: x[1], reverse=True)
    return [c for c, _ in shifts[:max_classes]]


def balanced_indices(labels, classes, per_class=50, random_state=0):
    rng = np.random.default_rng(random_state)
    idx_all = []

    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        take = min(len(idx), per_class)
        idx_all.extend(rng.choice(idx, size=take, replace=False).tolist())

    idx_all = np.asarray(idx_all, dtype=int)
    rng.shuffle(idx_all)
    return idx_all


def reduce_for_plot(
    features_a,
    features_b,
    labels,
    max_classes=20,
    per_class=50,
    reducer="umap",
    random_state=0,
):
    n = min(len(features_a), len(features_b), len(labels))
    features_a = features_a[:n]
    features_b = features_b[:n]
    labels = labels[:n]

    unique_classes = np.unique(labels)

    if len(unique_classes) > max_classes:
        classes = choose_plot_classes_by_shift(
            features_a,
            features_b,
            labels,
            max_classes=max_classes,
        )
    else:
        classes = unique_classes.tolist()

    idx = balanced_indices(labels, classes, per_class=per_class, random_state=random_state)

    fa = l2_normalize_np(features_a[idx].astype(np.float32))
    fb = l2_normalize_np(features_b[idx].astype(np.float32))
    y = labels[idx]

    x = np.concatenate([fa, fb], axis=0)

    pca_dim = min(50, x.shape[1], x.shape[0] - 1)
    if pca_dim >= 2:
        x0 = PCA(n_components=pca_dim, random_state=random_state).fit_transform(x)
    else:
        x0 = x

    if reducer == "umap" and umap is not None:
        emb = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            metric="cosine",
            random_state=random_state,
        ).fit_transform(x0)
        reducer_used = "umap"
    else:
        perplexity = min(30, max(5, (x0.shape[0] - 1) // 3))
        emb = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=random_state,
        ).fit_transform(x0)
        reducer_used = "tsne"

    emb_a = emb[:len(idx)]
    emb_b = emb[len(idx):]

    return emb_a, emb_b, y, classes, reducer_used


def save_feature_clustering_report(
    noadapt,
    adapted,
    output_dir,
    prefix="feature_cluster",
    reducer="umap",
    max_plot_classes=20,
    per_class=50,
    random_state=0,
):
    os.makedirs(output_dir, exist_ok=True)

    per_class = int(per_class)
    max_plot_classes = int(max_plot_classes)
    random_state = int(random_state)

    fa = noadapt["features"]
    fb = adapted["features"]
    labels = noadapt["labels"]

    n = min(len(fa), len(fb), len(labels))
    fa = fa[:n]
    fb = fb[:n]
    labels = labels[:n]

    np.savez_compressed(
        os.path.join(output_dir, f"{prefix}_features.npz"),
        noadapt_features=fa.astype(np.float32),
        adapted_features=fb.astype(np.float32),
        labels=labels.astype(np.int64),
        noadapt_preds=noadapt["preds"][:n].astype(np.int64),
        adapted_preds=adapted["preds"][:n].astype(np.int64),
        noadapt_confs=noadapt["confs"][:n].astype(np.float32),
        adapted_confs=adapted["confs"][:n].astype(np.float32),
        noadapt_logits=noadapt["logits"][:n].astype(np.float32),
        adapted_logits=adapted["logits"][:n].astype(np.float32),
    )

    metrics, per_class_metrics = compute_geometry_metrics(
    fa,
    fb,
    labels,
    random_state=random_state,
)

    meta = {
        "n_samples": int(n),
        "feature_layer_noadapt": noadapt.get("feature_layer", None),
        "feature_layer_adapted": adapted.get("feature_layer", None),
        "metrics": metrics,
        "per_class": per_class_metrics,
    }


    with open(os.path.join(output_dir, f"{prefix}_metrics.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    emb_a, emb_b, y, classes, reducer_used = reduce_for_plot(
        fa,
        fb,
        labels,
        max_classes=max_plot_classes,
        per_class=per_class,
        reducer=reducer,
        random_state=random_state,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharex=True, sharey=True)

    sc0 = axes[0].scatter(
        emb_a[:, 0],
        emb_a[:, 1],
        c=y,
        s=8,
        alpha=0.75,
        cmap="tab20",
        linewidths=0,
    )
    axes[0].set_title("No adaptation")
    axes[0].grid(True, alpha=0.15)

    sc1 = axes[1].scatter(
        emb_b[:, 0],
        emb_b[:, 1],
        c=y,
        s=8,
        alpha=0.75,
        cmap="tab20",
        linewidths=0,
    )
    axes[1].set_title("Adapted final")
    axes[1].grid(True, alpha=0.15)

    fig.suptitle(
        f"{reducer_used.upper()} on selected classes: {classes}\n"
        f"centroid ρ={metrics['centroid_distance_spearman']:.3f}, "
        f"sample ρ={metrics['sample_pairwise_distance_spearman']:.3f}"
    )

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_{reducer_used}.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Direct distance preservation plot: class centroid pairwise distances.
    classes_a, cent_a = class_centroids(l2_normalize_np(fa.astype(np.float32)), labels)
    classes_b, cent_b = class_centroids(l2_normalize_np(fb.astype(np.float32)), labels)
    common = np.intersect1d(classes_a, classes_b)

    if len(common) >= 3:
        ca = np.vstack([cent_a[np.where(classes_a == c)[0][0]] for c in common])
        cb = np.vstack([cent_b[np.where(classes_b == c)[0][0]] for c in common])

        da = pairwise_distances(ca, metric="cosine")
        db = pairwise_distances(cb, metric="cosine")
        tri = np.triu_indices_from(da, k=1)

        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        ax.scatter(da[tri], db[tri], s=10, alpha=0.65)
        lo = min(float(da[tri].min()), float(db[tri].min()))
        hi = max(float(da[tri].max()), float(db[tri].max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
        ax.set_xlabel("No-adapt centroid cosine distance")
        ax.set_ylabel("Adapted centroid cosine distance")
        ax.set_title("Class-centroid distance preservation")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{prefix}_centroid_distance_preservation.png"), dpi=220)
        plt.close(fig)

    return meta