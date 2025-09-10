import torch
import torch.nn as nn
from dataset import *
from model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import argparse
from einops import rearrange, repeat
from generate_loss import *

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def print_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} grad norm: {param.grad.norm().item():.4f}")
        else:
            print(f"{name} grad: None")


def training_VisNet_loss(pred_t, pred_beta, is_uniform_fog,
                  gt_t, gt_beta, foggy_img, clean_img, depth_map, epoch, mean_bgr, uniform_fog, need_det=True):
    pred_t = pred_t.clamp(min=0.01, max=1.0)
    t_recon = torch.exp(-pred_beta * depth_map)
    J_recon = (foggy_img - mean_bgr * (1 - pred_t)) / pred_t

    loss_b = F.l1_loss(pred_beta, gt_beta)
    loss_t = F.l1_loss(pred_t, gt_t)
    loss_ret = F.l1_loss(pred_t, t_recon)
    loss_reI = F.l1_loss(J_recon, clean_img)
    if epoch < 20:
        args.delta3 = 0
        args.delta4 = 0
    
    loss_vis = args.delta1 * loss_b + args.delta2 * loss_t + args.delta3 * loss_ret + args.delta4 * loss_reI
    if not need_det:
        return loss_vis

    loss_det = F.binary_cross_entropy_with_logits(is_uniform_fog, uniform_fog)
    total_loss = loss_vis * args.lambda1 + loss_det * args.lambda2
    return total_loss

def training_DMRVisNet_loss(pred_a, pred_t, pred_d, pred_defog, pred_vis, gt_t, gt_a, gt_mask, gt_d, clean_img, gt_v):
    mask = (gt_t < -1) | gt_mask
    loss_a = generate_loss("RMSE")(pred_a, gt_a)
    loss_t = generate_loss("RMSE")(pred_t, gt_t)
    loss_d = generate_loss("Reprojection")(pred_d, gt_d) + 0.001 * generate_loss("Smooth")(pred_d, clean_img)
    loss_defog = generate_loss("MaskedRMSE")(pred_defog, clean_img, mask)
    loss_vis = generate_loss("MaskedL1")(pred_vis, gt_v, mask)
    total_loss = loss_a + loss_t + 0.8 * loss_d + 1e-6 * loss_defog + loss_vis
    return total_loss

def train_VisNet_PD(train_item, model, optimizer, epoch, local_rank):
    clean_rgb = train_item["clean_rgb"]
    fog_rgb = train_item["fog_rgb"]
    vis = train_item["vis"]
    transmission_map = train_item["transmission_map"]
    metric_depth_map = train_item["metric_depth_map"]
    mean_bgr = train_item["mean_bgr"]
    uniform_fog = train_item["uniform_fog"]
    clean_rgb, fog_rgb, vis, transmission_map, metric_depth_map, mean_bgr, uniform_fog = clean_rgb.to(local_rank), fog_rgb.to(local_rank), vis.to(local_rank), transmission_map.to(local_rank), metric_depth_map.to(local_rank), mean_bgr.to(local_rank), uniform_fog.to(local_rank)
    beta = 2.995 / vis
    optimizer.zero_grad()
    t_output, beta_output, is_uniform_fog = model(fog_rgb, metric_depth_map)
    loss = training_VisNet_loss(t_output, beta_output, is_uniform_fog,
                            transmission_map, beta, fog_rgb, clean_rgb, metric_depth_map, epoch, mean_bgr, uniform_fog)
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return loss

def train_VisNet_FACID(train_item, model, optimizer, epoch, local_rank):
    clean_rgb = train_item["Scene"]
    fog_rgb = train_item["FoggyScene_0.05"]
    vis = train_item["Visibility"] / 1000
    transmission_map = train_item["t_0.05"]
    # ! FACID dataset provides inverse depth
    metric_depth_map = 1 / train_item["DepthPerspective"]
    metric_depth_map = torch.clamp(metric_depth_map, min=0, max=5e3) / 1000
    mean_bgr = train_item["A"]

    vis = repeat(vis, 'b 1 -> b 1 h w', h=fog_rgb.shape[2], w=fog_rgb.shape[3])
    beta = -math.log(0.05) / vis
    mean_bgr = rearrange(mean_bgr, 'b c -> b c 1 1')

    clean_rgb, fog_rgb, beta, transmission_map, metric_depth_map, mean_bgr = clean_rgb.to(local_rank), fog_rgb.to(local_rank), beta.to(local_rank), transmission_map.to(local_rank), metric_depth_map.to(local_rank), mean_bgr.to(local_rank)
    optimizer.zero_grad()
    t_output, beta_output = model(fog_rgb, metric_depth_map)
    loss = training_VisNet_loss(t_output, beta_output, None,
                            transmission_map, beta, fog_rgb, clean_rgb, metric_depth_map, epoch, mean_bgr, None, need_det=False)
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return loss

def train_DMRVisNet_PD(train_item, model, optimizer, epoch, local_rank):
    clean_rgb = train_item["clean_rgb"]
    fog_rgb = train_item["fog_rgb"]
    vis = train_item["vis"]
    transmission_map = train_item["transmission_map"]
    metric_depth_map = train_item["metric_depth_map"]
    mean_bgr = train_item["mean_bgr"]
    uniform_fog = train_item["uniform_fog"]
    clean_rgb, fog_rgb, vis, transmission_map, metric_depth_map, mean_bgr, uniform_fog = clean_rgb.to(local_rank), fog_rgb.to(local_rank), vis.to(local_rank), transmission_map.to(local_rank), metric_depth_map.to(local_rank), mean_bgr.to(local_rank), uniform_fog.to(local_rank)
    beta = 2.995 / vis
    optimizer.zero_grad()
    pred_a, pred_t, pred_d, pred_defog, pred_vis = model(fog_rgb)
    metric_depth_map = 1 / metric_depth_map
    loss = training_DMRVisNet_loss(pred_a, pred_t, pred_d, pred_defog, pred_vis,
                            transmission_map, mean_bgr, torch.zeros_like(transmission_map).bool().to(local_rank), metric_depth_map, clean_rgb, -beta)
    loss.backward()

    optimizer.step()
    return loss


def train_DMRVisNet_FACID(train_item, model, optimizer, epoch, local_rank):
    clean_rgb = train_item["Scene"]
    fog_rgb = train_item["FoggyScene_0.05"]
    vis = train_item["Visibility"] / 1000
    transmission_map = train_item["t_0.05"]
    metric_depth_map = train_item["DepthPerspective"]
    mask = train_item["SkyMask"]
    mean_bgr = train_item["A"]

    vis = repeat(vis, 'b 1 -> b 1 h w', h=fog_rgb.shape[2], w=fog_rgb.shape[3])
    beta = -math.log(0.05) / vis
    mean_bgr = rearrange(mean_bgr, 'b c -> b c 1 1')

    clean_rgb, fog_rgb, beta, transmission_map, metric_depth_map, mean_bgr, mask = clean_rgb.to(local_rank), fog_rgb.to(local_rank), beta.to(local_rank), transmission_map.to(local_rank), metric_depth_map.to(local_rank), mean_bgr.to(local_rank), mask.to(local_rank)
    optimizer.zero_grad()
    pred_a, pred_t, pred_d, pred_defog, pred_vis = model(fog_rgb)
    loss = training_DMRVisNet_loss(pred_a, pred_t, pred_d, pred_defog, pred_vis,
                            transmission_map, mean_bgr, mask, metric_depth_map, clean_rgb, -beta)
    loss.backward()
    optimizer.step()
    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--root_path", type=str)
    parser.add_argument("--dataset_type", type=str, default="PixelWise", choices=["PixelWise", "FACI"])
    parser.add_argument("--model_type", type=str, default="VisNet", choices=["VisNet", "DMRVisNet"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--cross_num", type=int, default=3)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--delta1", type=float, default=0.4)
    parser.add_argument("--delta2", type=float, default=1.0)
    parser.add_argument("--delta3", type=float, default=0.2)
    parser.add_argument("--delta4", type=float, default=0.1)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=0.4)

    args, _ = parser.parse_known_args()
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        setup(rank, world_size)
        print(f"Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")
        ddp = True
    except KeyError:
        rank = 0
        local_rank = f"cuda:{args.gpu_id}"
        world_size = 1
        ddp = False
        print("Running in single GPU mode")

    save_path = f"{args.save_path}_{args.model_type}_{args.dataset_type}"
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    if args.dataset_type == "PixelWise":
        train_dataset = PixelwiseDataset(args.root_path, phase='train', istrain=True)
        if args.model_type == "VisNet":
            train_function = train_VisNet_PD
        else:
            train_function = train_DMRVisNet_PD
    else:
        train_dataset = FACIDataset(args.root_path)
        if args.model_type == "VisNet":
            train_function = train_VisNet_FACID
        else:
            train_function = train_DMRVisNet_FACID
    if ddp:
        train_sampler = DistributedSampler(train_dataset,
                                           num_replicas=world_size,
                                           rank=local_rank,
                                           shuffle=True,
                                           drop_last=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                  shuffle=False, num_workers=4, drop_last=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    if args.model_type == "VisNet":
        model = VitNet(cross_num=args.cross_num, need_det=(args.dataset_type=="PixelWise"))
    else:
        model = DMRVisNet()
    
    model.to(local_rank)
    if ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    else:
        model = nn.DataParallel(model)
    if args.model_type == "VisNet":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 学习率衰减
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, weights_only=True)
        model_state_dict = checkpoint['model_state_dict']
        if args.model_type == "VisNet":
            keys = list(model_state_dict.keys())
            num_fusion_trans = sum(1 for k in keys if 'fusion_transformer' in k) // 14
            if num_fusion_trans != args.cross_num:
                for k in keys:
                    if 'fusion_transformer' in k:
                        model_state_dict.pop(k)
                model.load_state_dict(model_state_dict, False)
            else:
                model.load_state_dict(model_state_dict)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            model.load_state_dict(model_state_dict)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if args.resume:
        start_epoch = int(os.path.basename(args.checkpoint_path).replace("epoch", "").replace(".pth", ""))
        scheduler.last_epoch = start_epoch
    else:
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        if ddp:
            train_sampler.set_epoch(epoch)
        train_loader = tqdm(train_loader, desc=f'Epoch train {epoch+1}/{num_epochs}', leave=True)
        for batch_idx, train_item in enumerate(train_loader):
            loss = train_function(train_item, model, optimizer, epoch, local_rank)
            train_loss += loss.item()
            train_loader.set_postfix(loss=f'{loss.item():.4f}')
        train_loss /= len(train_loader)
        scheduler.step()  # 更新学习率
        os.makedirs(save_path, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
        }, os.path.join(save_path, f"epoch{epoch+1}.pth"))
    if ddp:
        dist.destroy_process_group()
