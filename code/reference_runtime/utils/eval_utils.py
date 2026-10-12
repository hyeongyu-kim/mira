import torch
import logging
import numpy as np
from typing import Union
from datasets.imagenet_subsets import IMAGENET_D_MAPPING
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.io import savemat   # ⬅️ 추가

import time
logger = logging.getLogger(__name__)


UNCERTAINTY_TRACE_BOOL_KEYS = [
    "performed_update",
    "collected_snapshot",
    "posterior_ready",
]

UNCERTAINTY_TRACE_INT_KEYS = [
    "posterior_n",
]

UNCERTAINTY_TRACE_FLOAT_KEYS = [
    "update_norm",
    "drift_norm",
    "drift_mean_norm",
    "residual_norm",
    "off_drift_ratio",
    "residual_mahalanobis_mean",
    "residual_mahalanobis_max",
    "residual_z_mean",
    "residual_orthogonal_ratio",
    "residual_gaussian_nll_mean",
    "residual_gaussian_nll_sum",
    "posterior_var_mean",
    "posterior_var_max",
    "uncertainty_score",
    # Label-free prediction reliability / risk. These never use labels.
    "base_confidence",
    "base_entropy",
    "base_entropy_norm",
    "confidence_margin",
    "posterior_pred_agreement",
    "posterior_pred_flip_rate",
    "posterior_pred_confidence",
    "posterior_pred_entropy",
    "posterior_pred_entropy_norm",
    "posterior_pred_mutual_info",
    "posterior_pred_mutual_info_norm",
    "posterior_pred_margin",
    "posterior_pred_logit_var_top1",
    "posterior_pred_samples",
    "virtual_update_kl",
    "virtual_update_kl_score",
    "virtual_top1_flip_rate",
    "virtual_conf_delta",
    "virtual_entropy_delta",
    "virtual_margin_delta",
    "label_free_risk",
    "label_free_reliability",
]

UNCERTAINTY_TRACE_KEYS = (
    UNCERTAINTY_TRACE_BOOL_KEYS
    + UNCERTAINTY_TRACE_INT_KEYS
    + UNCERTAINTY_TRACE_FLOAT_KEYS
)


def _prepare_targets(targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim > 1:
        targets = targets.argmax(dim=1)
    return targets.long()


def split_results_by_domain(domain_dict: dict, data: list, predictions: torch.tensor):
    """
    Separates the label prediction pairs by domain
    Input:
        domain_dict: Dictionary, where the keys are the domain names and the values are lists with pairs [[label1, prediction1], ...]
        data: List containing [images, labels, domains, ...]
        predictions: Tensor containing the predictions of the model
    Returns:
        domain_dict: Updated dictionary containing the domain seperated label prediction pairs
    """

    labels, domains = data[1], data[2]
    assert predictions.shape[0] == labels.shape[0], "The batch size of predictions and labels does not match!"

    for i in range(labels.shape[0]):
        if domains[i] in domain_dict.keys():
            domain_dict[domains[i]].append([labels[i].item(), predictions[i].item()])
        else:
            domain_dict[domains[i]] = [[labels[i].item(), predictions[i].item()]]

    return domain_dict


def eval_domain_dict(domain_dict: dict, domain_seq: list):
    """
    Print detailed results for each domain. This is useful for settings where the domains are mixed
    Input:
        domain_dict: Dictionary containing the labels and predictions for each domain
        domain_seq: Order to print the results (if all domains are contained in the domain dict)
    """
    correct = []
    num_samples = []
    avg_error_domains = []
    domain_names = domain_seq if all([dname in domain_seq for dname in domain_dict.keys()]) else domain_dict.keys()
    logger.info(f"Splitting the results by domain...")
    for key in domain_names:
        label_prediction_arr = np.array(domain_dict[key])  # rows: samples, cols: (label, prediction)
        correct.append((label_prediction_arr[:, 0] == label_prediction_arr[:, 1]).sum())
        num_samples.append(label_prediction_arr.shape[0])
        accuracy = correct[-1] / num_samples[-1]
        error = 1 - accuracy
        avg_error_domains.append(error)
        logger.info(f"{key:<20} error: {error:.2%}")
    logger.info(f"Average error across all domains: {sum(avg_error_domains) / len(avg_error_domains):.2%}")
    # The error across all samples differs if each domain contains different amounts of samples
    logger.info(f"Error over all samples: {1 - sum(correct) / sum(num_samples):.2%}")


def _to_prob_output(output: torch.Tensor, adaptation_name: str) -> torch.Tensor:
    # if ("_laplace" in adaptation_name) or (adaptation_name == "source_sample"):
    #     return output
    return torch.softmax(output, dim=1)


def _forward_model_for_eval(model, imgs, labels, batch_idx, device, cfg):
    """
    Rich trace kwargs(targets, batch_idx)를 지원하면 그 경로를 우선 사용.
    지원하지 않으면 기존 호출 방식으로 fallback.

    returns:
        output
        used_rich_forward: bool
    """
    adaptation_name = str(cfg.MODEL.ADAPTATION)
    labels_dev = labels.to(device)
    x_dev = [img.to(device) for img in imgs] if isinstance(imgs, list) else imgs.to(device)

    # if adaptation_name in ["tent_laplace", "source_sample"]:
    #     try:
    #         output = model(x_dev, labels_dev, batch_idx=batch_idx)
    #         return output, True
    #     except TypeError:
    output = model(x_dev)
    return output, False

    # try:
    #     output = model(x_dev, targets=labels_dev, batch_idx=batch_idx)
    #     return output, True
    # except TypeError:
    #     output = model(x_dev)
    #     return output, False
    


    

def summarize_update_trace_for_batch(model, labels_eval: torch.Tensor):
    """Compare selected update pseudo-labels with evaluation labels.

    Labels are used only here, after the label-free adaptation step, for diagnosis.
    If OPTIM.STEPS > 1, selected samples from all adaptation steps are pooled.
    """
    m = model.module if hasattr(model, "module") else model
    trace = getattr(m, "last_update_trace", None)
    steps = trace.get("steps", []) if isinstance(trace, dict) else []

    labels_cpu = labels_eval.detach().cpu().long()
    batch_size = int(labels_cpu.numel())

    total_selected = 0
    correct_values = []
    confidence_values = []

    for step in steps:
        selected_indices = step.get("selected_indices", torch.empty(0, dtype=torch.long)).long()
        selected_predictions = step.get("selected_predictions", torch.empty(0, dtype=torch.long)).long()
        selected_confidences = step.get("selected_confidences", torch.empty(0)).float()

        if selected_indices.numel() == 0:
            continue

        valid = (selected_indices >= 0) & (selected_indices < batch_size)
        selected_indices = selected_indices[valid]
        selected_predictions = selected_predictions[valid]
        selected_confidences = selected_confidences[valid]

        if selected_indices.numel() == 0:
            continue

        selected_labels = labels_cpu[selected_indices]
        correct_values.append((selected_predictions == selected_labels).float())
        confidence_values.append(selected_confidences)
        total_selected += int(selected_indices.numel())

    num_steps = max(len(steps), 1)
    denom = max(batch_size * num_steps, 1)

    if total_selected > 0:
        selected_correct = torch.cat(correct_values)
        selected_conf = torch.cat(confidence_values)
        selected_acc = float(selected_correct.mean().item())
        selected_conf_mean = float(selected_conf.mean().item())
    else:
        selected_acc = float("nan")
        selected_conf_mean = float("nan")

    return {
        "update_selected_acc": selected_acc,
        "update_selected_conf": selected_conf_mean,
        "update_selected_count": total_selected,
        "update_selected_ratio": float(total_selected / denom),
        "update_num_steps": int(len(steps)),
    }

def get_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_name: str,
                 domain_name: str,
                 setting: str,
                 domain_dict: dict,
                 print_every: int,
                 device: Union[str, torch.device],
                 return_probs=False,
                 return_online_trace=False,
                 cfg=None,
                 adapt_until_batch=None):

    num_correct = 0.
    num_samples = 0

    all_probs = []
    all_targets = []

    online_trace = {
    "batch_acc": [],
    "batch_conf": [],
    "batch_mmc": [],
    "batch_ece": [],
    "batch_ce": [],
    "batch_entropy": [],
    "pred_batch_index": [],
    "predicted_class": [],
    "batch_size": [],
    "adapt_enabled": [],
    }

    for key in UNCERTAINTY_TRACE_KEYS:
        online_trace[key] = []
    

    check_update = bool(getattr(cfg.TEST, "CHECK_UPDATE", False)) if cfg is not None else False
    if check_update:
        online_trace.update({
            "update_selected_acc": [],
            "update_selected_conf": [],
            "update_selected_count": [],
            "update_selected_ratio": [],
            "update_num_steps": [],
        })

    print(len(data_loader), " loader_len")

    t1 = time.time()
    with torch.no_grad():
        for i, data in enumerate(data_loader):


            m_for_adapt = model.module if hasattr(model, "module") else model
            adapt_enabled = True

            if adapt_until_batch is not None:
                adapt_enabled = i < adapt_until_batch
                if hasattr(m_for_adapt, "set_adaptation_enabled"):
                    m_for_adapt.set_adaptation_enabled(adapt_enabled)


            imgs, labels = data[0], data[1]

            output, used_rich_forward = _forward_model_for_eval(
                model=model,
                imgs=imgs,
                labels=labels,
                batch_idx=i,
                device=device,
                cfg=cfg,
            )

            adaptation_name = str(cfg.MODEL.ADAPTATION)
            probs = _to_prob_output(output, adaptation_name)

            # rich-forward를 아직 안 붙인 방법만 evaluator 쪽 legacy logging 사용
            if not used_rich_forward:
                m = model.module if hasattr(model, "module") else model

                labels_for_log = labels.to(device)
                if labels_for_log.ndim > 1:
                    labels_for_log = labels_for_log.argmax(dim=1)
                labels_for_log = labels_for_log.long()

                if hasattr(m, "record_online_eval"):
                    m.record_online_eval(
                        probs.detach(),
                        labels_for_log.detach(),
                        batch_idx=i,
                    )

            raw_predictions = probs.argmax(1)
            confidences = probs.max(1).values

            labels_eval = labels.to(device)
            if labels_eval.ndim > 1:
                labels_eval = labels_eval.argmax(dim=1)
            labels_eval = labels_eval.long()

            predictions = raw_predictions
            if dataset_name == "imagenet_d" and domain_name != "none":
                mapping_vector = list(IMAGENET_D_MAPPING.values())
                predictions = torch.tensor(
                    [mapping_vector[pred] for pred in raw_predictions],
                    device=device,
                )

            batch_correct = (predictions == labels_eval).float()
            num_correct += batch_correct.sum()

            if return_online_trace:
                batch_mmc = confidences.mean().item()
                batch_ece = compute_ece(
                    probs.detach(),
                    labels_eval.detach(),
                    num_bins=int(getattr(cfg.TEST, "METRIC_NUM_BINS", 10)),
                )
                online_trace["batch_acc"].append(batch_correct.mean().item())
                online_trace["batch_conf"].append(batch_mmc)
                online_trace["batch_mmc"].append(batch_mmc)
                online_trace["batch_ece"].append(batch_ece)
                online_trace["pred_batch_index"].extend([i] * raw_predictions.numel())
                online_trace["predicted_class"].extend(raw_predictions.detach().cpu().tolist())
                online_trace["batch_size"].append(int(raw_predictions.numel()))
                online_trace["adapt_enabled"].append(bool(adapt_enabled))

                unc = getattr(m, "last_uncertainty", {}) if hasattr(m, "last_uncertainty") else {}
                for key in UNCERTAINTY_TRACE_BOOL_KEYS:
                    online_trace[key].append(bool(unc.get(key, False)))
                for key in UNCERTAINTY_TRACE_INT_KEYS:
                    online_trace[key].append(int(unc.get(key, 0)))
                for key in UNCERTAINTY_TRACE_FLOAT_KEYS:
                    value = unc.get(key, float("nan"))
                    try:
                        online_trace[key].append(float(value))
                    except (TypeError, ValueError):
                        online_trace[key].append(float("nan"))

                if check_update:
                    update_summary = summarize_update_trace_for_batch(model, labels_eval)
                    online_trace["update_selected_acc"].append(update_summary["update_selected_acc"])
                    online_trace["update_selected_conf"].append(update_summary["update_selected_conf"])
                    online_trace["update_selected_count"].append(update_summary["update_selected_count"])
                    online_trace["update_selected_ratio"].append(update_summary["update_selected_ratio"])
                    online_trace["update_num_steps"].append(update_summary["update_num_steps"])

                batch_ce = F.nll_loss(
                    torch.log(probs.detach().clamp_min(1e-12)),
                    labels_eval.detach(),
                    reduction="mean",
                ).item()

                batch_entropy = (
                    -(probs.detach() * torch.log(probs.detach().clamp_min(1e-12)))
                    .sum(dim=1)
                    .mean()
                    .item()
                )

                online_trace["batch_ce"].append(batch_ce)
                online_trace["batch_entropy"].append(batch_entropy)

                # trace_arrays["batch_ce"] = np.asarray(trace.get("batch_ce", []), dtype=float)
                # trace_arrays["batch_entropy"] = np.asarray(trace.get("batch_entropy", []), dtype=float)




            if "mixed_domains" in setting and len(data) >= 3:
                domain_dict = split_results_by_domain(domain_dict, data, predictions)

            num_samples += imgs[0].shape[0] if isinstance(imgs, list) else imgs.shape[0]
            if print_every > 0 and (i + 1) % print_every == 0:
                logger.info(
                    f"#batches={i+1:<6} #samples={num_samples:<9} "
                    f"error = {1 - num_correct / num_samples:.2%}"
                )

            if dataset_name == "ccc" and num_samples >= 7500000:
                break

            all_probs.append(probs.detach().cpu())
            all_targets.append(labels.detach().cpu())

    accuracy = num_correct.item() / num_samples

    t2 = time.time()
    # print("TIME : ", t2 - t1, "SIGMA: ", cfg.SIG_SCALE, " mC : ", cfg.NUM_MC)

    m_for_adapt = model.module if hasattr(model, "module") else model
    if adapt_until_batch is not None and hasattr(m_for_adapt, "set_adaptation_enabled"):
        m_for_adapt.set_adaptation_enabled(True)

    if return_probs:
        all_probs_cat = torch.cat(all_probs, dim=0)
        all_targets_cat = torch.cat(all_targets, dim=0)
        if return_online_trace:
            return accuracy, domain_dict, num_samples, all_probs_cat, all_targets_cat, online_trace
        return accuracy, domain_dict, num_samples, all_probs_cat, all_targets_cat
    if return_online_trace:
        return accuracy, domain_dict, num_samples, online_trace
    return accuracy, domain_dict, num_samples
    






def compute_metrics(all_probs: torch.Tensor, all_targets: torch.Tensor, num_bins: int = 10):
    """
    전체 데이터에 대해 한 번에 메트릭 계산 (첫 번째 코드 방식)
    Input:
        all_probs: 모든 샘플의 예측 확률 (N, num_classes)
        all_targets: 모든 샘플의 실제 레이블 (N,)
        num_bins: ECE 계산을 위한 bin 개수
    Returns:
        metrics: 계산된 메트릭 딕셔너리
    """
    metrics = {}
    all_targets = _prepare_targets(all_targets)
    
    # MMC (Maximum Mean Confidence)
    c, preds = torch.max(all_probs, 1)
    metrics['mmc'] = c.mean().item()
    
    # Accuracy
    metrics['acc'] = (all_targets == preds).float().mean().item()
    
    loss_fn = torch.nn.NLLLoss()
    # NLL (Negative Log-Likelihood)
    log_probs = torch.log(torch.clamp(all_probs, min=1e-12, max=1.0))
    metrics['nll'] = loss_fn(log_probs, all_targets.long()).item()
    
    # ECE (Expected Calibration Error)
    metrics['ece'] = compute_ece(all_probs, all_targets, num_bins=num_bins)
    
    # Brier Score
    metrics['brier'] = compute_brier(all_probs, all_targets)
    
    # AUROC
    try:
        probs_np = all_probs.numpy()
        targets_np = all_targets.numpy()
        auroc_list = []
        for c_idx in range(probs_np.shape[1]):
            y_true = (targets_np == c_idx).astype(int)
            if len(np.unique(y_true)) > 1:
                auroc_c = roc_auc_score(y_true, probs_np[:, c_idx])
                auroc_list.append(auroc_c)
        metrics['auroc'] = np.mean(auroc_list) if auroc_list else np.nan
    except Exception as e:
        logger.warning(f"AUROC calculation failed: {e}")
        metrics['auroc'] = np.nan
    
    return metrics


def compute_ece(all_probs: torch.Tensor, all_targets: torch.Tensor, num_bins: int = 10) -> float:
    """
    전체 데이터에 대해 ECE 계산 (첫 번째 코드 방식 - get_calib과 동일)
    """
    all_targets = _prepare_targets(all_targets)

    # Get max confidence and predictions
    confidences, predictions = torch.max(all_probs, dim=-1)
    accuracies = (predictions == all_targets).float()
    
    # Convert to numpy for binning
    confs_np = confidences.cpu().numpy()
    accs_np = accuracies.cpu().numpy()
    
    # Create bins
    bins = np.linspace(0, 1, num_bins + 1)
    
    # Accuracy and avg. confidence per bin
    accs_bin = []
    confs_bin = []
    nitems_bin = []
    
    for i in range(num_bins):
        # Find samples in this bin
        if i == num_bins - 1:  # Last bin includes right edge
            bin_mask = (confs_np >= bins[i]) & (confs_np <= bins[i + 1])
        else:
            bin_mask = (confs_np >= bins[i]) & (confs_np < bins[i + 1])
        
        accs_i = accs_np[bin_mask]
        confs_i = confs_np[bin_mask]
        
        if len(accs_i) > 0:
            acc = np.mean(accs_i)
            conf = np.mean(confs_i)
            nitems = len(accs_i)
        else:
            acc = 0.0
            conf = 0.0
            nitems = 0
        
        accs_bin.append(acc)
        confs_bin.append(conf)
        nitems_bin.append(nitems)
    
    accs_bin = np.array(accs_bin)
    confs_bin = np.array(confs_bin)
    nitems_bin = np.array(nitems_bin)
    
    # Calculate ECE (weighted average)
    total_items = nitems_bin.sum()
    if total_items > 0:
        ece = np.sum(np.abs(confs_bin - accs_bin) * nitems_bin) / total_items
    else:
        ece = 0.0
    
    return float(ece)


def compute_brier(all_probs: torch.Tensor, all_targets: torch.Tensor):
    """
    전체 데이터에 대해 Brier Score 계산 (첫 번째 코드 방식 - get_brier_score와 동일)
    """
    num_classes = all_probs.shape[1]
    targets_onehot = F.one_hot(all_targets.long(), num_classes=num_classes).float()
    
    # Move to same device as probs
    if all_probs.device != targets_onehot.device:
        targets_onehot = targets_onehot.to(all_probs.device)
    
    # Brier score: mean squared error (mean over samples, sum over classes)
    brier = torch.mean(torch.sum((all_probs - targets_onehot) ** 2, dim=1)).item()
    
    return brier
