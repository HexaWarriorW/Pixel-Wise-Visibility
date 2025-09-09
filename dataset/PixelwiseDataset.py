import os
import pickle
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
from torchvision.transforms import functional as F

class PixelwiseDataset(Dataset):
    def __init__(self, path, istrain = True, isVis = False):
        self.filenames = os.listdir(path)
        self.filenames = [os.path.join(path, file) for file in self.filenames]
        self.istrain = istrain
        self.isVis = isVis
        self.mean = [0.63233793, 0.62714498, 0.61009363]
        self.std = [0.20397434, 0.20220747, 0.21616411]

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        with open(self.filenames[idx], 'rb') as f:
            file_pkl = pickle.load(f)
        fog_rgb = file_pkl["fog_rgb"]
        metric_depth_map = file_pkl["metric_depth_map"]
        clean_rgb = file_pkl["clean_rgb"]
        vis = file_pkl["vis"]
        uniform_fog = file_pkl["uniform_fog"]
        mean_bgr = file_pkl["mean_bgr"]
        if self.istrain:
            clean_rgb, fog_rgb, metric_depth_map, vis = self.vis_transform(clean_rgb,fog_rgb,metric_depth_map,vis, self.isVis)
        else:
            clean_rgb, fog_rgb, metric_depth_map, vis = self.vis_nontransform(clean_rgb,fog_rgb,metric_depth_map,vis, self.isVis)
        beta = 2.995 / vis
        transmission_map = np.exp(-beta * metric_depth_map)
        data_item = dict(
            clean_rgb = clean_rgb,
            fog_rgb = fog_rgb,
            vis = vis,
            transmission_map = torch.from_numpy(transmission_map).unsqueeze(0).float(),
            metric_depth_map = metric_depth_map,
            mean_bgr = torch.from_numpy(mean_bgr).float(),
            uniform_fog = torch.from_numpy(uniform_fog).unsqueeze(0).float(),
        )
        return data_item

    def vis_transform(self, clean_rgb, fog_rgb, depth_img, vis, isVis):
        vis = Image.fromarray(vis.astype(np.float32))
        depth_img = Image.fromarray(depth_img.astype(np.float32))

        color_jitter = transforms.ColorJitter(
            brightness=0.1, 
            contrast=0.15,  
            saturation=0.15,
            hue=0.05,       
        )
        
        if random.random() > 0.5:
            clean_rgb = F.hflip(clean_rgb)
            fog_rgb = F.hflip(fog_rgb)
            depth_img = F.hflip(depth_img)
            vis = F.hflip(vis)
        if random.random() > 0.5:    
            clean_rgb = F.vflip(clean_rgb)
            fog_rgb = F.vflip(fog_rgb)
            depth_img = F.vflip(depth_img)
            vis = F.vflip(vis)

        fog_rgb = transforms.ToTensor()(fog_rgb)
        fog_rgb = F.resize(fog_rgb, (800, 800),antialias=False)
        i, j, h, w = transforms.RandomCrop.get_params(fog_rgb, (640, 640))
        fog_rgb = F.crop(fog_rgb, i, j, h, w)
        if not isVis:
            fog_rgb = color_jitter(fog_rgb)
            fog_rgb = transforms.Normalize(mean=self.mean, std=self.std)(fog_rgb)
        clean_rgb = transforms.ToTensor()(clean_rgb)
        clean_rgb = F.resize(clean_rgb, (800, 800),antialias=False)
        clean_rgb = F.crop(clean_rgb, i, j, h, w)
        if not isVis:
            clean_rgb = transforms.Normalize(mean=self.mean, std=self.std)(clean_rgb)

        depth_img = transforms.ToTensor()(depth_img)
        depth_img = F.resize(depth_img, (800, 800),antialias=False)
        depth_img = F.crop(depth_img, i, j, h, w)
        vis = transforms.ToTensor()(vis)
        vis = F.resize(vis, (800, 800),antialias=False)
        vis = F.crop(vis, i, j, h, w)

        return clean_rgb, fog_rgb, depth_img, vis

    def vis_nontransform(self, clean_rgb, rgb_img, depth_img, vis, isVis):
        depth_img = Image.fromarray(depth_img.astype(np.float32))
        vis = Image.fromarray(vis.astype(np.float32))

        rgb_img = transforms.ToTensor()(rgb_img)
        rgb_img = F.resize(rgb_img, (640, 640),antialias=False)
        if not isVis:
            rgb_img = transforms.Normalize(mean=self.mean, std=self.std)(rgb_img)

        clean_rgb = transforms.ToTensor()(clean_rgb)
        clean_rgb = F.resize(clean_rgb, (640, 640),antialias=False)
        if not isVis:
            clean_rgb = transforms.Normalize(mean=self.mean, std=self.std)(clean_rgb)

        depth_img = transforms.ToTensor()(depth_img)
        depth_img = F.resize(depth_img, (640, 640),antialias=False)

        vis = transforms.ToTensor()(vis)
        vis = F.resize(vis, (640, 640),antialias=False)
        return clean_rgb, rgb_img, depth_img, vis


if __name__ == "__main__":
    root_path = "/media/arthurthomas/CN300/vis/fog_dataset/val/"
    val_dataset = PixelwiseDataset(root_path, istrain=False, isVis=True)

    batch_size = 1
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    for batch_idx, (clean_rgb, fog_rgb, vis, transmission_map, metric_depth_map, mean_bgr, uniform_fog) in enumerate(val_loader):
        clean_rgb = (clean_rgb.permute(0, 2, 3, 1)[0].numpy() * 255).astype(np.uint8)
        fog_rgb = (fog_rgb.permute(0, 2, 3, 1)[0].numpy() * 255).astype(np.uint8)
        clean_rgb = cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2BGR)
        fog_rgb = cv2.cvtColor(fog_rgb, cv2.COLOR_RGB2BGR)
        vis = vis.permute(0, 2, 3, 1)[0].numpy()
        transmission_map = (transmission_map.permute(0, 2, 3, 1)[0].numpy() * 255).astype(np.uint8)
        metric_depth_map = metric_depth_map.permute(0, 2, 3, 1)[0].numpy()
        metric_depth_map = np.log(metric_depth_map)
        metric_depth_map = (metric_depth_map - metric_depth_map.min()) / (metric_depth_map.max() - metric_depth_map.min() + 1e-8)
        metric_depth_map = (metric_depth_map * 255).astype(np.uint8)
        metric_depth_map = cv2.applyColorMap(metric_depth_map, cv2.COLORMAP_INFERNO)
        transmission_map = cv2.applyColorMap(transmission_map, cv2.COLORMAP_INFERNO)
        vis = 3 / vis
        vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
        vis = (vis * 255).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
        cv2.namedWindow("clean_rgb", 0)
        cv2.imshow("clean_rgb", clean_rgb)
        cv2.namedWindow("fog_rgb", 0)
        cv2.imshow("fog_rgb", fog_rgb)
        cv2.namedWindow("vis", 0)
        cv2.imshow("vis", vis)
        cv2.namedWindow("transmission_map", 0)
        cv2.imshow("transmission_map", transmission_map)
        cv2.namedWindow("metric_depth_map", 0)
        cv2.imshow("metric_depth_map", metric_depth_map)

        while True:
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('e'):
                exit(0)
