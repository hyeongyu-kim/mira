"""The paper's 10-bin equal-width calibration metrics, with accuracy in percent."""
import torch


def evaluate(probabilities, labels):
    p = probabilities.detach().cpu()
    y = labels.detach().long().cpu()
    confidence,prediction = p.max(-1)
    correct = (prediction==y).float()
    ece = 0.
    for i in range(10):
        mask = (confidence>=i/10) & ((confidence<(i+1)/10) if i<9 else (confidence<=1))
        if mask.any():
            ece += float(mask.float().mean()*abs(confidence[mask].mean()-correct[mask].mean()))
    return dict(accuracy_percent=100*float(correct.mean()), mmc=float(confidence.mean()),
                ece=ece, nll=float(-p.clamp_min(1e-12).log()[torch.arange(len(y)),y].mean()))
