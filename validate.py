import numpy as np
import torch
import torch.nn as nn
from dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from metric import *
from einops import repeat
from model import *
import argparse
from thop import profile as t_profile

def val_VisNet_PD(val_item, model, device):
    fog_rgb = val_item["fog_rgb"]
    vis = val_item["vis"]
    transmission_map = val_item["transmission_map"]
    metric_depth_map = val_item["metric_depth_map"]
    uniform_fog = val_item["uniform_fog"]

    fog_rgb, metric_depth_map = fog_rgb.to(device), metric_depth_map.to(device)
    t_output, beta_output, is_uniform_fog = model(fog_rgb, metric_depth_map)
    t_output = t_output.detach().cpu().numpy()
    beta_output = beta_output.detach().cpu().numpy()
    is_uniform_fog = nn.Sigmoid()(is_uniform_fog).detach().cpu().numpy()

    metric_depth_map = metric_depth_map.cpu().numpy()
    v_output = 2.995 / (beta_output + 1e-6)

    transmission_map = transmission_map.numpy()
    vis = vis.numpy()

    v_output_pred, _ = calc_vis(v_output)
    v_output_gt = np.mean(vis)
    # in meters
    v_output_pred *= 1000
    v_output_gt *= 1000
    return v_output_pred, v_output_gt, uniform_fog.numpy(), is_uniform_fog

def val_VisNet_FACID(val_item, model, device):
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
    t_output = t_output.detach().cpu().numpy()
    beta_output = beta_output.detach().cpu().numpy()

    metric_depth_map = metric_depth_map.cpu().numpy()
    v_output = 2.995 / (beta_output + 1e-6)

    transmission_map = transmission_map.numpy()
    vis = vis.numpy()

    v_output_pred, _ = calc_vis(v_output)
    v_output_gt = np.mean(vis.reshape(vis.shape[0], -1), axis=1)
    # in meters
    v_output_pred *= 1000
    v_output_gt *= 1000
    return v_output_pred, v_output_gt, 0, 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str)
    parser.add_argument("--dataset_type", type=str, default="PixelWise", choices=["PixelWise", "FACI"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cross_num", type=int, default=3)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--cal_flops", action="store_true")
    args, _ = parser.parse_known_args()
    cal_flops = args.cal_flops
    if args.dataset_type == "PixelWise":
        val_dataset = PixelwiseDataset(args.root_path, phase='val', istrain=False)
        val_function = val_VisNet_PD
    else:
        val_dataset = FACIDataset(args.root_path, phase='valid')
        val_function = val_VisNet_FACID
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    model = VitNet(cross_num=args.cross_num, need_det=(args.dataset_type=="PixelWise"))
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    val_loader = tqdm(val_loader, leave=True)
    with torch.no_grad():
        errors = np.zeros((len(val_loader) * args.batch_size, 3))
        for batch_idx, val_item in enumerate(val_loader):
            if cal_flops:
                fog_rgb = val_item['fog_rgb'].to(device)
                metric_depth_map = val_item['metric_depth_map'].to(device)
                fog_rgb, metric_depth_map = fog_rgb.to(device), metric_depth_map.to(device)
                flops, params = t_profile(model, (fog_rgb, metric_depth_map))
                print(f'FLOPs = {flops / 1000**3:.3f}G')
                print(f'Params = {params / 1000**2:.3f}M')
                cal_flops = False
            v_output_pred, v_output_gt, uniform_fog, is_uniform_fog = val_function(val_item, model, device)
            absrel_val = AbsRel(v_output_pred, v_output_gt)
            sqrel_val = SqRel(v_output_pred, v_output_gt)
            rmse_val = rmse(v_output_pred, v_output_gt)
            errors[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size] = np.stack([absrel_val, sqrel_val, rmse_val], 1)

        error_mean = errors.mean(axis=0)
        print(f"cross_num {args.cross_num} AbsRel: {error_mean[0]:.3f}, SqRel: {error_mean[1]:.3f}, RMSE: {error_mean[2]:.3f}")