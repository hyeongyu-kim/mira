import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def _unwrap_tta_model(model):
    return model.module if hasattr(model, "module") else model


def _move_to_device(x, device):
    if isinstance(x, (list, tuple)):
        return [xx.to(device, non_blocking=True) if torch.is_tensor(xx) else xx for xx in x]
    return x.to(device, non_blocking=True) if torch.is_tensor(x) else x


def _first_view(imgs):
    return imgs[0] if isinstance(imgs, (list, tuple)) else imgs


@torch.no_grad()
def _flat_params(tta):
    if hasattr(tta, "_flat_landscape_params"):
        return tta._flat_landscape_params()

    return torch.cat([p.detach().float().cpu().reshape(-1) for p in tta.params])


@torch.no_grad()
def _assign_flat_params(tta, vec):
    if hasattr(tta, "_assign_landscape_params"):
        tta._assign_landscape_params(vec)
        return

    offset = 0
    for p in tta.params:
        n = p.numel()
        chunk = vec[offset:offset + n].view_as(p).to(device=p.device, dtype=p.dtype)
        p.copy_(chunk)
        offset += n

def _compute_multi_projection_and_grid(
    tta,
    base_model,
    runs,
    batches,
    grid_size=25,
    pad=0.35,
    ref_mode="source",
):
    """
    runs: list of dicts
      {
        "name": "tent",
        "trajectory": Tensor[T, D],
        "online_trace": dict or None,
      }
    """
    trajectories = []
    names = []

    for r in runs:
        traj = r["trajectory"]
        if traj is None or traj.shape[0] < 2:
            continue
        trajectories.append(traj.float().cpu())
        names.append(r["name"])

    if len(trajectories) == 0:
        return None

    dim = trajectories[0].shape[1]
    for name, traj in zip(names, trajectories):
        if traj.shape[1] != dim:
            raise ValueError(
                f"Trajectory dim mismatch for {name}: {traj.shape[1]} vs {dim}. "
                "Use LANDSCAPE_PARAM_MODE='all_norm'."
            )

    if ref_mode == "source":
        ref = trajectories[0][0].clone()
    elif ref_mode == "mean":
        ref = torch.cat(trajectories, dim=0).mean(dim=0)
    else:
        raise ValueError(f"Unknown ref_mode={ref_mode}")

    centered_all = torch.cat([traj - ref for traj in trajectories], dim=0)

    if float(centered_all.norm()) < 1e-12:
        return None

    _, _, vh = torch.linalg.svd(centered_all, full_matrices=False)
    if vh.shape[0] < 2:
        return None

    v1 = vh[0] / vh[0].norm().clamp_min(1e-12)
    v2 = vh[1] / vh[1].norm().clamp_min(1e-12)
    dirs = torch.stack([v1, v2], dim=1)

    run_coords = {}
    all_coords = []

    for r, traj in zip(runs, trajectories):
        coords = (traj - ref) @ dirs
        coords_np = coords.numpy()
        run_coords[r["name"]] = coords_np
        all_coords.append(coords_np)

    all_coords = np.concatenate(all_coords, axis=0)

    x_min, x_max = float(all_coords[:, 0].min()), float(all_coords[:, 0].max())
    y_min, y_max = float(all_coords[:, 1].min()), float(all_coords[:, 1].max())

    x_rad = max(abs(x_min), abs(x_max), 1e-3)
    y_rad = max(abs(y_min), abs(y_max), 1e-3)

    x_min, x_max = x_min - pad * x_rad, x_max + pad * x_rad
    y_min, y_max = y_min - pad * y_rad, y_max + pad * y_rad

    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)

    final_vec = _flat_params(tta)
    module_modes = [(m, m.training) for m in base_model.modules()]

    z_ce = np.zeros((len(xs), len(ys)), dtype=np.float32)
    z_ent = np.zeros((len(xs), len(ys)), dtype=np.float32)

    try:
        base_model.eval()

        for i, a in enumerate(xs):
            for j, b in enumerate(ys):
                vec = ref + float(a) * v1 + float(b) * v2
                _assign_flat_params(tta, vec)

                z_ce[i, j] = _landscape_loss(tta, base_model, batches, "ce")
                z_ent[i, j] = _landscape_loss(tta, base_model, batches, "entropy")

    finally:
        _assign_flat_params(tta, final_vec)
        for module, was_training in module_modes:
            module.train(was_training)

    return {
        "ref": ref,
        "v1": v1,
        "v2": v2,
        "xs": xs,
        "ys": ys,
        "z_ce": z_ce,
        "z_ent": z_ent,
        "run_coords": run_coords,
    }


def _collect_batches(data_loader, device, max_batches):
    batches = []
    for i, data in enumerate(data_loader):
        if i >= max_batches:
            break

        imgs = _move_to_device(data[0], device)
        labels = data[1].to(device, non_blocking=True) if len(data) > 1 and torch.is_tensor(data[1]) else None
        batches.append((imgs, labels))

    return batches


@torch.no_grad()
def _landscape_loss(tta, base_model, batches, loss_type):
    vals = []

    for imgs, labels in batches:
        x = _first_view(imgs)
        logits = base_model(x)

        if loss_type == "ce":
            if labels is None:
                raise ValueError("LANDSCAPE_LOSS='ce' requires labels.")
            y = labels
            if y.ndim > 1:
                y = y.argmax(dim=1)
            loss = F.cross_entropy(logits, y.long())

        elif loss_type == "entropy":
            probs = torch.softmax(logits, dim=1)
            loss = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1).mean()

        else:
            raise ValueError(f"Unknown loss_type={loss_type}")

        vals.append(float(loss.detach().cpu()))

    return float(np.mean(vals))


@torch.no_grad()
def _metrics_on_batches(base_model, batches):
    total_correct = 0.0
    total_num = 0
    ce_vals = []
    ent_vals = []

    for imgs, labels in batches:
        x = _first_view(imgs)
        logits = base_model(x)
        probs = torch.softmax(logits, dim=1)

        ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1).mean()
        ent_vals.append(float(ent.detach().cpu()))

        if labels is not None:
            y = labels
            if y.ndim > 1:
                y = y.argmax(dim=1)
            y = y.long()

            ce = F.cross_entropy(logits, y)
            ce_vals.append(float(ce.detach().cpu()))

            preds = probs.argmax(dim=1)
            total_correct += float((preds == y).float().sum().item())
            total_num += int(y.numel())

    acc = (total_correct / total_num) if total_num > 0 else np.nan
    ce_mean = float(np.mean(ce_vals)) if len(ce_vals) > 0 else np.nan
    ent_mean = float(np.mean(ent_vals)) if len(ent_vals) > 0 else np.nan

    return {
        "acc": acc,
        "ce": ce_mean,
        "entropy": ent_mean,
    }


def _trajectory_fixed_metrics(tta, base_model, trajectory, batches):
    final_vec = _flat_params(tta)
    module_modes = [(m, m.training) for m in base_model.modules()]

    accs, ces, ents = [], [], []

    try:
        base_model.eval()
        for vec in trajectory:
            _assign_flat_params(tta, vec)
            out = _metrics_on_batches(base_model, batches)
            accs.append(out["acc"])
            ces.append(out["ce"])
            ents.append(out["entropy"])
    finally:
        _assign_flat_params(tta, final_vec)
        for module, was_training in module_modes:
            module.train(was_training)

    return np.asarray(accs), np.asarray(ces), np.asarray(ents)


def _compute_projection_and_grid(tta, base_model, trajectory, batches, grid_size=25, pad=0.35):
    final_vec = _flat_params(tta)
    trajectory = trajectory.float().cpu()

    ref = trajectory.mean(dim=0)
    centered = trajectory - ref

    if float(centered.norm()) < 1e-12:
        return None

    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    if vh.shape[0] < 2:
        return None

    v1 = vh[0] / vh[0].norm().clamp_min(1e-12)
    v2 = vh[1] / vh[1].norm().clamp_min(1e-12)

    dirs = torch.stack([v1, v2], dim=1)
    coords = centered @ dirs
    coords_np = coords.numpy()

    x_min, x_max = float(coords_np[:, 0].min()), float(coords_np[:, 0].max())
    y_min, y_max = float(coords_np[:, 1].min()), float(coords_np[:, 1].max())

    x_rad = max(abs(x_min), abs(x_max), 1e-3)
    y_rad = max(abs(y_min), abs(y_max), 1e-3)

    x_min, x_max = x_min - pad * x_rad, x_max + pad * x_rad
    y_min, y_max = y_min - pad * y_rad, y_max + pad * y_rad

    if abs(x_max - x_min) < 1e-6:
        x_min, x_max = -1.0, 1.0
    if abs(y_max - y_min) < 1e-6:
        y_min, y_max = -1.0, 1.0

    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)

    module_modes = [(m, m.training) for m in base_model.modules()]
    z_ce = np.zeros((len(xs), len(ys)), dtype=np.float32)
    z_ent = np.zeros((len(xs), len(ys)), dtype=np.float32)

    try:
        base_model.eval()
        for i, a in enumerate(xs):
            for j, b in enumerate(ys):
                vec = ref + float(a) * v1 + float(b) * v2
                _assign_flat_params(tta, vec)
                z_ce[i, j] = _landscape_loss(tta, base_model, batches, "ce")
                z_ent[i, j] = _landscape_loss(tta, base_model, batches, "entropy")
    finally:
        _assign_flat_params(tta, final_vec)
        for module, was_training in module_modes:
            module.train(was_training)

    return {
        "ref": ref,
        "coords_np": coords_np,
        "xs": xs,
        "ys": ys,
        "z_ce": z_ce,
        "z_ent": z_ent,
    }

def save_multi_tta_landscape_panel(
    tta_model,
    data_loader,
    runs,
    output_dir,
    prefix,
    device,
    grid_size=25,
    max_batches=5,
    pad=0.35,
    ref_mode="source",
):
    """
    Plot multiple TTA trajectories on the same pooled-PCA landscape.

    runs:
      [
        {"name": "tent", "trajectory": tensor, "online_trace": dict},
        {"name": "eata", "trajectory": tensor, "online_trace": dict},
        {"name": "deyo", "trajectory": tensor, "online_trace": dict},
      ]
    """
    tta = _unwrap_tta_model(tta_model)
    base_model = getattr(tta, "model", tta)

    batches = _collect_batches(data_loader, device, max_batches=max_batches)
    if len(batches) == 0:
        return None

    proj = _compute_multi_projection_and_grid(
        tta=tta,
        base_model=base_model,
        runs=runs,
        batches=batches,
        grid_size=grid_size,
        pad=pad,
        ref_mode=ref_mode,
    )
    if proj is None:
        return None

    xs = proj["xs"]
    ys = proj["ys"]
    z_ce = proj["z_ce"]
    z_ent = proj["z_ent"]
    run_coords = proj["run_coords"]

    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(18.5, 7.2))
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[1.0, 1.0],
        wspace=0.28,
    )

    preferred_colors = {
        "tent":  "#1f77b4",  # blue
        "eata":  "#d62728",  # red
        "deyo":  "#2ca02c",  # green
        "sar":   "#9467bd",  # purple
        "roid":  "#ff7f0e",  # orange
        "cotta": "#8c564b",  # brown
        "eta":   "#e377c2",  # pink
        "norm":  "#7f7f7f",  # gray
    }

    palette = list(plt.get_cmap("tab20").colors)

    method_names = [r["name"].lower() for r in runs if r["name"] in run_coords]
    used = set()
    colors = {}

    # 먼저 선호 색 배정
    for name in method_names:
        if name in preferred_colors:
            colors[name] = preferred_colors[name]
            used.add(preferred_colors[name])

    # 없는 method는 남는 palette에서 자동 배정
    palette_idx = 0
    for name in method_names:
        if name in colors:
            continue
        while palette_idx < len(palette) and palette[palette_idx] in used:
            palette_idx += 1
        if palette_idx < len(palette):
            colors[name] = palette[palette_idx]
            used.add(palette[palette_idx])
            palette_idx += 1
        else:
            colors[name] = None

    def _plot_trajectories(ax):
        for r in runs:
            name = r["name"]
            if name not in run_coords:
                continue

            coords = run_coords[name]
            c = colors.get(name.lower(), None)

            ax.plot(
                coords[:, 0],
                coords[:, 1],
                marker=".",
                linewidth=1.7,
                markersize=4,
                color=c,
                label=name,
            )
            ax.scatter(
                [coords[0, 0]],
                [coords[0, 1]],
                marker="o",
                s=45,
                color=c,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.scatter(
                [coords[-1, 0]],
                [coords[-1, 1]],
                marker="x",
                s=80,
                color=c,
                linewidth=2.0,
            )

    # entropy landscape
    ax0 = fig.add_subplot(gs[0, 0])
    contour0 = ax0.contourf(xs, ys, z_ent.T, levels=30)
    ax0.contour(xs, ys, z_ent.T, levels=12, linewidths=0.45, alpha=0.6)
    _plot_trajectories(ax0)
    ax0.set_title("Entropy landscape: multi-method trajectories")
    ax0.set_xlabel("pooled PC1")
    ax0.set_ylabel("pooled PC2")
    ax0.grid(True, alpha=0.15)
    ax0.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=max(1, min(len(runs), 5)),
        fontsize=9,
        framealpha=0.95,
    )
    fig.colorbar(contour0, ax=ax0, fraction=0.046, pad=0.04)

    # CE landscape
    ax1 = fig.add_subplot(gs[0, 1])
    contour1 = ax1.contourf(xs, ys, z_ce.T, levels=30)
    ax1.contour(xs, ys, z_ce.T, levels=12, linewidths=0.45, alpha=0.6)
    _plot_trajectories(ax1)
    ax1.set_title("CE landscape: multi-method trajectories")
    ax1.set_xlabel("pooled PC1")
    ax1.set_ylabel("pooled PC2")
    ax1.grid(True, alpha=0.15)
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=max(1, min(len(runs), 5)),
        fontsize=9,
        framealpha=0.95,
    )
    fig.colorbar(contour1, ax=ax1, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = os.path.join(output_dir, f"{prefix}_multi_landscape_panel.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        os.path.join(output_dir, f"{prefix}_multi_landscape_panel.npz"),
        xs=xs,
        ys=ys,
        ce_loss=z_ce,
        entropy_loss=z_ent,
        **{f"{name}_xy": coords for name, coords in run_coords.items()},
    )

    return path



def save_tta_landscape_panel(
    tta_model,
    data_loader,
    output_dir,
    prefix,
    device,
    grid_size=25,
    max_batches=5,
    pad=0.35,
    online_trace=None,
):
    """
    Final layout:
      col0: entropy landscape
      col1: CE landscape
      col2(top): online batch metrics
      col2(bot): fixed-subset metrics along trajectory
    """
    tta = _unwrap_tta_model(tta_model)
    base_model = getattr(tta, "model", tta)

    if not hasattr(tta, "get_landscape_trajectory"):
        return None

    trajectory = tta.get_landscape_trajectory()
    if trajectory is None or trajectory.shape[0] < 3:
        return None

    final_vec = _flat_params(tta)
    if not torch.allclose(trajectory[-1], final_vec, atol=1e-7, rtol=1e-5):
        trajectory = torch.cat([trajectory, final_vec[None]], dim=0)

    batches = _collect_batches(data_loader, device, max_batches=max_batches)
    if len(batches) == 0:
        return None

    proj = _compute_projection_and_grid(
        tta=tta,
        base_model=base_model,
        trajectory=trajectory,
        batches=batches,
        grid_size=grid_size,
        pad=pad,
    )
    if proj is None:
        return None

    coords_np = proj["coords_np"]
    xs = proj["xs"]
    ys = proj["ys"]
    z_ce = proj["z_ce"]
    z_ent = proj["z_ent"]

    fixed_acc, fixed_ce, fixed_ent = _trajectory_fixed_metrics(
        tta, base_model, trajectory, batches
    )

    os.makedirs(output_dir, exist_ok=True)

    # fixed colors
    ACC_COLOR = "tab:blue"
    CE_COLOR = "tab:red"
    ENT_COLOR = "tab:orange"
    TRAJ_COLOR = "tab:cyan"

    # much larger figure
    fig = plt.figure(figsize=(20.0, 8.8))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.15, 1.15, 1.05],
        height_ratios=[1.0, 1.0],
        wspace=0.38,
        hspace=0.35,
    )

    # ---------------------------------------------------
    # Left: entropy landscape
    # ---------------------------------------------------
    ax0 = fig.add_subplot(gs[:, 0])
    contour0 = ax0.contourf(xs, ys, z_ent.T, levels=30)
    ax0.contour(xs, ys, z_ent.T, levels=12, linewidths=0.45, alpha=0.6)

    ax0.plot(coords_np[:, 0], coords_np[:, 1], color=TRAJ_COLOR, marker=".", linewidth=1.5, label="Trajectory (proj)")
    ax0.scatter([0.0], [0.0], color=TRAJ_COLOR, marker="*", s=180, label="Trajectory mean")
    ax0.scatter([coords_np[0, 0]], [coords_np[0, 1]], color="tab:orange", marker="o", s=55, label="start")
    ax0.scatter([coords_np[-1, 0]], [coords_np[-1, 1]], color="tab:green", marker="x", s=70, label="final")

    ax0.set_xlabel("PC1")
    ax0.set_ylabel("PC2")
    ax0.set_title("Tent entropy loss landscape")
    ax0.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=9, framealpha=0.95)
    fig.colorbar(contour0, ax=ax0, fraction=0.046, pad=0.04)

    # ---------------------------------------------------
    # Middle: CE landscape
    # ---------------------------------------------------
    ax1 = fig.add_subplot(gs[:, 1])
    contour1 = ax1.contourf(xs, ys, z_ce.T, levels=30)
    ax1.contour(xs, ys, z_ce.T, levels=12, linewidths=0.45, alpha=0.6)

    ax1.plot(coords_np[:, 0], coords_np[:, 1], color=TRAJ_COLOR, marker=".", linewidth=1.5, label="Trajectory (proj)")
    ax1.scatter([0.0], [0.0], color=TRAJ_COLOR, marker="*", s=180, label="Trajectory mean")
    ax1.scatter([coords_np[0, 0]], [coords_np[0, 1]], color="tab:orange", marker="o", s=55, label="start")
    ax1.scatter([coords_np[-1, 0]], [coords_np[-1, 1]], color="tab:green", marker="x", s=70, label="final")

    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title("Tent CE loss landscape")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=9, framealpha=0.95)
    fig.colorbar(contour1, ax=ax1, fraction=0.046, pad=0.04)

    # ---------------------------------------------------
    # Right-top: online batch metrics
    # ---------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2])
    ax2b = ax2.twinx()

    if online_trace is not None and len(online_trace.get("batch_acc", [])) > 0:
        batch_x = np.arange(len(online_trace["batch_acc"]))
        lines = []
        labels = []

        l = ax2.plot(
            batch_x,
            np.asarray(online_trace["batch_acc"], dtype=float),
            color=ACC_COLOR,
            linewidth=1.5,
            label="Accuracy",
        )
        lines += l
        labels += ["Accuracy"]

        if "batch_ce" in online_trace and len(online_trace["batch_ce"]) == len(batch_x):
            l = ax2b.plot(
                batch_x,
                np.asarray(online_trace["batch_ce"], dtype=float),
                color=CE_COLOR,
                linewidth=1.5,
                label="CE",
            )
            lines += l
            labels += ["CE"]

        if "batch_entropy" in online_trace and len(online_trace["batch_entropy"]) == len(batch_x):
            l = ax2b.plot(
                batch_x,
                np.asarray(online_trace["batch_entropy"], dtype=float),
                color=ENT_COLOR,
                linewidth=1.5,
                linestyle="--",
                alpha=0.95,
                label="Entropy",
            )
            lines += l
            labels += ["Entropy"]

        ax2.set_xlabel("Online batch")
        ax2.set_ylabel("Accuracy", color=ACC_COLOR)
        ax2.tick_params(axis="y", labelcolor=ACC_COLOR)
        ax2.set_ylim(0.0, 1.0)

        ax2b.set_ylabel("Loss")
        ax2.set_title("Online batch metrics")
        ax2.grid(True, alpha=0.25)

        # move legend above plot so it does not cover data
        ax2.legend(
            lines, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            ncol=3,
            fontsize=9,
            framealpha=0.95,
        )
    else:
        ax2.text(0.5, 0.5, "No online_trace provided", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Online batch metrics")

    # ---------------------------------------------------
    # Right-bottom: fixed-subset metrics
    # ---------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 2])
    ax3b = ax3.twinx()

    step_x = np.arange(len(fixed_acc))
    lines = []
    labels = []

    if np.isfinite(fixed_acc).any():
        l = ax3.plot(
            step_x,
            fixed_acc,
            color=ACC_COLOR,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label="Fixed-subset acc",
        )
        lines += l
        labels += ["Fixed-subset acc"]

    if np.isfinite(fixed_ce).any():
        l = ax3b.plot(
            step_x,
            fixed_ce,
            color=CE_COLOR,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label="Fixed-subset CE",
        )
        lines += l
        labels += ["Fixed-subset CE"]

    if np.isfinite(fixed_ent).any():
        l = ax3b.plot(
            step_x,
            fixed_ent,
            color=ENT_COLOR,
            marker="o",
            markersize=3,
            linewidth=1.5,
            linestyle="--",
            alpha=0.95,
            label="Fixed-subset entropy",
        )
        lines += l
        labels += ["Fixed-subset entropy"]

    ax3.set_xlabel("Trajectory checkpoint")
    ax3.set_ylabel("Accuracy", color=ACC_COLOR)
    ax3.tick_params(axis="y", labelcolor=ACC_COLOR)
    ax3.set_ylim(0.0, 1.0)
    ax3b.set_ylabel("Loss")
    ax3.set_title("Fixed-subset metrics along trajectory")
    ax3.grid(True, alpha=0.25)

    # legend outside
    ax3.legend(
        lines, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        fontsize=9,
        framealpha=0.95,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    path = os.path.join(output_dir, f"{prefix}_landscape_panel.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        os.path.join(output_dir, f"{prefix}_landscape_panel.npz"),
        xs=xs,
        ys=ys,
        ce_loss=z_ce,
        entropy_loss=z_ent,
        trajectory_xy=coords_np,
        fixed_acc=fixed_acc,
        fixed_ce=fixed_ce,
        fixed_entropy=fixed_ent,
    )

    return path