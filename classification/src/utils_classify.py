import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

import utils as u

plt.ioff()

def calculate_metrics(predictions, targets, threshold = 0.5):
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    # Convert predictions to binary labels based on threshold
    pred_labels = (predictions >= threshold).astype(int)
    true_labels = targets
    # # Convert one-hot encoded targets to class labels

    # Replace NaNs with zeros
    predictions = np.nan_to_num(predictions)
    targets = np.nan_to_num(targets)
    
    # Precision and Recall for each class
    precision_macro = metrics.precision_score(true_labels, pred_labels, average='macro')
    recall_macro = metrics.recall_score(true_labels, pred_labels, average='macro')
    f1_score_macro = metrics.f1_score(true_labels, pred_labels, average='macro')
    map_macro = metrics.average_precision_score(targets, predictions, average='macro')

    precision_micro = metrics.precision_score(true_labels, pred_labels, average='micro')
    recall_micro = metrics.recall_score(true_labels, pred_labels, average='micro')
    f1_score_micro = metrics.f1_score(true_labels, pred_labels, average='micro')
    map_micro = metrics.average_precision_score(targets, predictions, average='micro')
    
    return {
            'precision_macro': precision_macro, 'recall_macro': recall_macro, 'f1_macro': f1_score_macro,
            'precision_micro': precision_micro, 'recall_micro': recall_micro, 'f1_micro': f1_score_micro,
            'map_macro': map_macro, 'map_micro': map_micro, 
            }


def ind2multihot(inds, num_labels):
    multihot = torch.zeros(num_labels)
    multihot = multihot.index_fill(0, torch.Tensor(inds).to(torch.int64), 1)
    return multihot

def logit2multihot(logits, threshold=0.5):
    return (logits > threshold).long()

def plot_performance(csv_path, start_step=0, title=None, save=True):
    if save:
        plt.ioff()
    keys = [
        "trn_loss", 
        "val_loss", 
        ]
    data = u.read_csv(csv_path, numeric=True)
    x_lr_changes = []
    vals = {key: {"x":[], "y":[]} for key in keys}
    old_lr = data[0]["lr"]
    for item in data:
        step = item["step"]
        if step >= start_step:
            new_lr = item["lr"]
            if new_lr < old_lr:
                x_lr_changes.append(step)
                old_lr = new_lr

            for key in keys:
                val = item[key]
                if not np.isnan(val):
                    vals[key]["x"].append(step)
                    vals[key]["y"].append(val)
    plt.figure(dpi=300)
    for key, points in vals.items():
        plt.plot(points["x"], points["y"], label=key)

    label = f"LR changes (x{len(x_lr_changes)})"
    for x in x_lr_changes:
        plt.axvline(x=x, color="black", linestyle="--", linewidth=1, label=label)
        label = None
    plt.legend()
    plt.grid()
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    if title == None:
        title = csv_path.split("/")[-2]
    plt.title(title)
    png_path = csv_path.replace(".csv", ".pdf")
    if save:
        plt.savefig(png_path)
    else:
        plt.show()
    plt.close()

