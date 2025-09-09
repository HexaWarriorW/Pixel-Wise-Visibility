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

def training_loss(pred_t, pred_beta, transmission_map, foggy_img, clean_img, depth_map, epoch, mean_bgr):
    t_recon = torch.exp(-pred_beta * depth_map)
    loss_t_consist = F.l1_loss(pred_t, t_recon)
    loss_t = F.l1_loss(pred_t, transmission_map)

    pred_t = pred_t.clamp(min=0.01, max=1.0)
    J_recon = (foggy_img - mean_bgr * (1 - pred_t)) / pred_t
    loss_recon = F.l1_loss(J_recon, clean_img)
    if epoch < 20:
        total_loss = loss_t + 0 * loss_t_consist + 0 * loss_recon
    else:
        total_loss = loss_t + 0.2 * loss_t_consist + 0.1 * loss_recon
    return total_loss*10

def train_patchy_loss(pred_t, gt):
    loss = F.binary_cross_entropy_with_logits(pred_t, gt)
    return loss