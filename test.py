import cv2
import numpy as np
import torch
import torch.nn as nn
from dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from model import VitResNet50
from einops import repeat
from metric import *
import argparse
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 25
plt.rcParams['axes.unicode_minus'] = False

def create_histgram_plot(data, pred, gt, gt_list, dpi=100):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    weights, bins, patches = ax.hist(data, bins=500, range=(100, 4999), alpha=0.8)
    max_height = max(weights)
    ax.fill_between([gt_list[0], gt_list[1]], 0, max_height, color='r', alpha=0.3)
    ax.vlines(gt, 0, max_height, colors='r', linestyles='dashed', label=f'GT={gt:.3f}m')
    ax.vlines(pred, 0, max_height, colors='b', linestyles='dashed', label=f'Pred={pred:.3f}m')
    ax.set_xlabel("Visibility Range(m)")
    ax.yaxis.set_visible(False)
    plt.tight_layout()
    plt.legend()
    plt.show()

# TODO: ONLY SUPPORT BATCH SIZE 1
def test_PD(val_item, model, device, batch_idx):
    fog_rgb = val_item["fog_rgb"]
    vis = val_item["vis"]
    transmission_map = val_item["transmission_map"]
    metric_depth_map = val_item["metric_depth_map"]
    uniform_fog = val_item["uniform_fog"]

    fog_rgb, metric_depth_map = fog_rgb.to(device), metric_depth_map.to(device)
    t_output, beta_output, is_uniform_fog = model(fog_rgb, metric_depth_map)
    t_output = t_output.detach().cpu().numpy()[0, 0, :, :]
    beta_output = beta_output.detach().cpu().numpy()[0, 0, :, :]
    is_uniform_fog = nn.Sigmoid()(is_uniform_fog).detach().cpu().numpy()

    metric_depth_map = metric_depth_map.cpu().numpy()[0, 0, :, :]
    v_output = 2.995 / (beta_output + 1e-3)

    transmission_map = transmission_map.numpy()[0, 0, :, :]
    vis = vis.numpy()[0, 0, :, :]

    # Post Process
    v_output_pred, v_output_hist = calc_vis(v_output)
    v_output_pred = v_output_pred[0]
    v_output_hist = v_output_hist[0]
    v_output_gt = np.mean(vis)
    if args.vis:
        vis_function(v_output_hist, v_output_pred, v_output_gt, fog_rgb, t_output, v_output, transmission_map, metric_depth_map, vis, batch_idx)

    return v_output_pred, v_output_gt, uniform_fog.item(), is_uniform_fog.item()


# TODO: ONLY SUPPORT BATCH SIZE 1
def test_FACID(val_item, model, device, batch_idx):
    fog_rgb = val_item["FoggyScene_0.05"]
    vis = val_item["Visibility"]
    transmission_map = val_item["t_0.05"]
    # ! FACID dataset provides inverse depth
    metric_depth_map = 1 / val_item["DepthPerspective"]
    vis = repeat(vis, 'b 1 -> b 1 h w', h=fog_rgb.shape[2], w=fog_rgb.shape[3])

    metric_depth_map = torch.clamp(metric_depth_map, min=0, max=5e3) / 1000
    vis = vis / 1000

    fog_rgb, metric_depth_map = fog_rgb.to(device), metric_depth_map.to(device)
    t_output, beta_output = model(fog_rgb, metric_depth_map)
    t_output = t_output.detach().cpu().numpy()[0, 0, :, :]
    beta_output = beta_output.detach().cpu().numpy()[0, 0, :, :]

    metric_depth_map = metric_depth_map.cpu().numpy()[0, 0, :, :]
    v_output = 2.995 / (beta_output + 1e-3)

    transmission_map = transmission_map.numpy()[0, 0, :, :]
    vis = vis.numpy()[0, 0, :, :]

    # Post Process
    v_output_pred, v_output_hist = calc_vis(v_output)
    v_output_pred = v_output_pred[0]
    v_output_hist = v_output_hist[0]
    v_output_gt = np.mean(vis)
    if args.vis:
        vis_function(v_output_hist, v_output_pred, v_output_gt, fog_rgb, t_output, v_output, transmission_map, metric_depth_map, vis, batch_idx)

    return v_output_pred, v_output_gt, 0, 0

def vis_function(v_output_hist, v_output_pred, v_output_gt, fog_rgb, t_output, v_output, transmission_map, metric_depth_map, vis, batch_idx):
    create_histgram_plot(v_output_hist.flatten()*1000, v_output_pred*1000, v_output_gt*1000, [np.max(vis)*1000, np.min(vis)*1000])
    fog_rgb_vis = fog_rgb.detach().cpu().permute(0, 2, 3, 1)[0].numpy()
    fog_rgb_vis = fog_rgb_vis * val_dataset.std + val_dataset.mean
    fog_rgb_vis = (fog_rgb_vis * 255).astype(np.uint8)
    fog_rgb_vis = cv2.cvtColor(fog_rgb_vis, cv2.COLOR_RGB2BGR)

    metric_depth_map_vis = np.log(metric_depth_map)
    metric_depth_map_vis = (metric_depth_map_vis - metric_depth_map_vis.min()) / (metric_depth_map_vis.max() - metric_depth_map_vis.min() + 1e-8)
    metric_depth_map_vis = (metric_depth_map_vis * 255).astype(np.uint8)

    t_output_vis = (t_output * 255).astype(np.uint8)
    v_output_vis = np.clip(v_output, 0, 5) / 30
    v_output_vis = (v_output_vis * 255).astype(np.uint8)
    vis_vis = np.clip(vis, 0, 5) / 30
    vis_vis = (vis_vis * 255).astype(np.uint8)

    transmission_map_vis = (transmission_map * 255).astype(np.uint8)

    transmission_map_vis = cv2.applyColorMap(transmission_map_vis, cv2.COLORMAP_INFERNO)
    t_output_vis = cv2.applyColorMap(t_output_vis, cv2.COLORMAP_INFERNO)
    v_output_vis = cv2.applyColorMap(v_output_vis, cv2.COLORMAP_INFERNO)
    vis_vis = cv2.applyColorMap(vis_vis, cv2.COLORMAP_INFERNO)

    cv2.namedWindow("clean_rgb", 0)
    cv2.imshow("clean_rgb", fog_rgb_vis)
    cv2.namedWindow("t_output_vis", 0)
    cv2.imshow("t_output_vis", t_output_vis)
    cv2.namedWindow("transmission_map_vis", 0)
    cv2.imshow("transmission_map_vis", transmission_map_vis)
    cv2.namedWindow("v_output_vis", 0)
    cv2.imshow("v_output_vis", v_output_vis)
    cv2.namedWindow("vis_vis", 0)
    cv2.imshow("vis_vis", vis_vis)

    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('e'):
            exit(0)
        elif key == ord('s'):
            os.makedirs(os.path.join(args.save_path, f"{batch_idx}"), exist_ok=True)
            cv2.imwrite(os.path.join(args.save_path, f"{batch_idx}/clean_rgb.png"), fog_rgb_vis)
            cv2.imwrite(os.path.join(args.save_path, f"{batch_idx}/transmission_map_vis.png"), transmission_map_vis)
            cv2.imwrite(os.path.join(args.save_path, f"{batch_idx}/v_output_vis.png"), v_output_vis)
            plt.savefig(os.path.join(args.save_path, f"{batch_idx}/violin_plot.png"), dpi=200)
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--root_path", type=str)
    parser.add_argument("--dataset_type", type=str, default="PixelWise", choices=["PixelWise", "FACI"])
    parser.add_argument("--cross_num", type=int, default=3)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--vis", action="store_true")
    args, _ = parser.parse_known_args()
    if args.dataset_type == "PixelWise":
        val_dataset = PixelwiseDataset(args.root_path, istrain=False)
        val_function = test_PD
    else:
        val_dataset = FACIDataset(args.root_path, phase='test')
        val_function = test_FACID

    plt.ion()
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)
    model = VitResNet50(cross_num=args.cross_num, need_det=(args.dataset_type=="PixelWise"))
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    model.eval()
    v_output_gt_list = []
    v_output_pred_list = []
    val_loader = tqdm(val_loader, leave=True)
    with torch.no_grad():
        for batch_idx, val_item in enumerate(val_loader):
            v_output_pred, v_output_gt, uniform_fog, is_uniform_fog = val_function(val_item, model, device, batch_idx)
            val_loader.set_postfix(
                vg = f"{v_output_gt:.3f}", 
                vp = f"{v_output_pred:.3f}",
                ug = f"{uniform_fog:.3f}",
                up = f"{is_uniform_fog:.3f}"
            )
            v_output_gt_list.append(v_output_gt)
            v_output_pred_list.append(v_output_pred)
            
