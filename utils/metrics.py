import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from monai.metrics import SurfaceDistanceMetric, HausdorffDistanceMetric, DiceMetric, ConfusionMatrixMetric
import torch.nn.functional as F

def multi_class_score_batched(metric_cls, predictions, labels, num_classes, spacing=None, **metric_kwargs):
    device = predictions.device
    results = torch.zeros(num_classes, device=device)
    metric = metric_cls(**metric_kwargs)

    for class_idx in range(num_classes):
        pred_bin = (predictions == class_idx).float().unsqueeze(1)
        label_bin = (labels == class_idx).float().unsqueeze(1)

        def run_metric():

            if spacing is not None and hasattr(metric, "_compute_tensor") and "spacing" in metric._compute_tensor.__code__.co_varnames:
                metric(pred_bin, label_bin, spacing=spacing)
            else:
                metric(pred_bin, label_bin)

            agg = metric.aggregate()
            return torch.stack(agg).mean() if isinstance(agg, list) else agg.mean()

        results[class_idx] = run_metric()

    return results.detach().cpu().numpy()


def hausdorff_distance(predictions, labels, num_classes, spacing=[1, 1, 1]):
    return multi_class_score_batched(
        HausdorffDistanceMetric,
        predictions,
        labels,
        num_classes,
        spacing=spacing,
        include_background=True,
        percentile=100.0
    )

def average_surface_distance(predictions, labels, num_classes, spacing=[1, 1, 1]):
    return multi_class_score_batched(
        SurfaceDistanceMetric,
        predictions,
        labels,
        num_classes,
        spacing=spacing,
        include_background=True,
        symmetric=True,
        distance_metric="euclidean"
    )

def one_sided_surface_distance(predictions, labels, num_classes, spacing=[1, 1, 1]):
    """
    Measures one-sided surface distance from labels → predictions for each class.
    """
    return multi_class_score_batched(
        SurfaceDistanceMetric,
        labels,
        predictions,
        num_classes,
        spacing=spacing,
        include_background=True,
        symmetric=False,
        distance_metric="euclidean"
    )

def dice_score(predictions, labels, num_classes):
    """
    Computes Dice score per class using MONAI, GPU-accelerated.

    Args:
        predictions: [B, C, D, H, W] one-hot or softmaxed probabilities
        labels: [B, 1, D, H, W] integer class indices (or one-hot if preprocessed)

    Returns:
        Tensor [num_classes] with average dice scores per class
    """
    if labels.shape[1] == 1:
        labels = torch.nn.functional.one_hot(labels.squeeze(1).long(), num_classes=num_classes)  # [B, D, H, W, C]
        labels = labels.permute(0, 4, 1, 2, 3).float()  # [B, C, D, H, W]

    return multi_class_score_batched(
        DiceMetric,
        predictions,
        labels,
        num_classes,
        include_background=True,
        reduction="mean"
    )

def precision(predictions, labels, num_classes):
    """
    Computes per-class precision using MONAI's ConfusionMatrixMetric.
    
    Args:
        predictions: [B, D, H, W] with integer class predictions
        labels: [B, D, H, W] with integer class labels
        num_classes: number of classes

    Returns:
        Tensor [num_classes] with precision per class
    """
    return multi_class_score_batched(
        ConfusionMatrixMetric,
        predictions,
        labels,
        num_classes,
        metric_name="precision",
        include_background=True
    )


def recall(predictions, labels, num_classes):
    """
    Computes per-class recall using MONAI's ConfusionMatrixMetric.

    Args:
        predictions: [B, D, H, W] with integer class predictions
        labels: [B, D, H, W] with integer class labels
        num_classes: number of classes

    Returns:
        Tensor [num_classes] with recall per class
    """
    return multi_class_score_batched(
        ConfusionMatrixMetric,
        predictions,
        labels,
        num_classes,
        metric_name="recall",
        include_background=True
    )


class Logger():
    def __init__(self, name, loss_names):
        self.name = name
        self.loss_names = loss_names
        self.epoch_logger = {}
        self.epoch_summary = {}
        self.epoch_number_logger = []
        self.reset_epoch_logger()
        self.reset_epoch_summary()

    def reset_epoch_logger(self):
        for loss_name in self.loss_names:
            self.epoch_logger[loss_name] = []

    def reset_epoch_summary(self):
        for loss_name in self.loss_names:
            self.epoch_summary[loss_name] = []

    def update_epoch_logger(self, loss_dict):
        for loss_name, loss_value in loss_dict.items():
            if loss_name not in self.loss_names:
                raise ValueError('Logger was not constructed to log {}'.format(loss_name))
            else:
                self.epoch_logger[loss_name].append(loss_value)

    def update_epoch_summary(self, epoch, reset=True):
        for loss_name in self.loss_names:
            self.epoch_summary[loss_name].append(np.mean(self.epoch_logger[loss_name]))
        self.epoch_number_logger.append(epoch)
        if reset:
            self.reset_epoch_logger()

    def get_latest_dict(self):
        latest = {}
        for loss_name in self.loss_names:
            latest[loss_name] = self.epoch_summary[loss_name][-1]
        return latest

    def get_epoch_logger(self):
        return self.epoch_logger

    def write_epoch_logger(self, location, index, loss_names, loss_labels, colours, linestyles=None, scales=None,
                           clear_plot=True):
        if linestyles is None:
            linestyles = ['-'] * len(colours)
        if scales is None:
            scales = [1] * len(colours)
        if not (len(loss_names) == len(loss_labels) and len(loss_labels) == len(colours) and len(colours) == len(
                linestyles) and len(linestyles) == len(scales)):
            raise ValueError('Length of all arg lists must be equal but got {} {} {} {} {}'.format(len(loss_names),
                                                                                                   len(loss_labels),
                                                                                                   len(colours),
                                                                                                   len(linestyles),
                                                                                                   len(scales)))

        for name, label, colour, linestyle, scale in zip(loss_names, loss_labels, colours, linestyles, scales):
            if scale == 1:
                plt.plot(range(0, len(self.epoch_logger[name])), self.epoch_logger[name], c=colour,
                         label=label, linestyle=linestyle)
            else:
                plt.plot(range(0, len(self.epoch_logger[name])), [scale * val for val in self.epoch_logger[name]],
                         c=colour,
                         label='{} x {}'.format(scale, label), linestyle=linestyle)
        plt.legend(loc='upper right')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig('{}/{}.png'.format(location, index))
        if clear_plot:
            plt.clf()

    def print_latest(self, loss_names=None):
        if not self.epoch_number_logger:
            return
        print_str = '{}\tEpoch: {}\t'.format(self.name, self.epoch_number_logger[-1])
        if loss_names is None:
            loss_names = self.loss_names
        for loss_name in loss_names:
            if loss_name not in self.loss_names:
                raise ValueError('Logger was not constructed to log {}'.format(loss_name))
            else:
                print_str += '{}: {:.6f}\t'.format(loss_name, self.epoch_summary[loss_name][-1])
        print(print_str)
