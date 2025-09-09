import torch
import torch.nn.functional as F
import numpy as np

def AbsRel(pred, gt):
    B = pred.shape[0]
    errors = []
    for b in range(B):
        valid_mask = (gt[b] > 0.01)
        error = np.mean(np.abs(pred[b][valid_mask] - gt[b][valid_mask]) / (gt[b][valid_mask] + 1e-6))
        errors.append(error)
    return np.array(errors)

def SqRel(pred, gt):
    B = pred.shape[0]
    errors = []
    for b in range(B):
        valid_mask = (gt[b] > 0.01)
        error = np.mean(((pred[b][valid_mask] - gt[b][valid_mask]) ** 2) / (gt[b][valid_mask] ** 2 + 1e-6))
        errors.append(error)
    return np.array(errors)

def rmse(pred, gt):
    B = pred.shape[0]
    errors = []
    for b in range(B):
        valid_mask = (gt[b] > 0.01)
        error = np.sqrt(np.mean((pred[b][valid_mask] - gt[b][valid_mask]) ** 2))
        errors.append(error)
    return np.array(errors)


def compute_confusion_matrix_elements(pred, gt):
    TP = np.sum((gt == 1) & (pred == 1))
    TN = np.sum((gt == 0) & (pred == 0))
    FP = np.sum((gt == 0) & (pred == 1))
    FN = np.sum((gt == 1) & (pred == 0))

    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }

def precision(confusion_matrix):
    TP = confusion_matrix['TP']
    FP = confusion_matrix['FP']
    return TP / (TP + FP + 1e-6)

def recall(confusion_matrix):
    TP = confusion_matrix['TP']
    FN = confusion_matrix['FN']
    return TP / (TP + FN + 1e-6)

def accuracy(confusion_matrix):
    TP = confusion_matrix['TP']
    TN = confusion_matrix['TN']
    FP = confusion_matrix['FP']
    FN = confusion_matrix['FN']
    return (TP + TN) / (TP + TN + FP + FN + 1e-6)


def print_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} grad norm: {param.grad.norm().item():.4f}")
        else:
            print(f"{name} grad: None")
