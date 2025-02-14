import os

import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from data_set import SpectrogramSet
import numpy as np


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()

    with open(path, 'wb') as f:
        np.save(f, ndarr)
    # ndarr = (ndarr * 255).astype(np.uint8)
    # im = Image.fromarray(ndarr)
    # im.save(path)


def get_data(args):
    # transforms = torchvision.transforms.Compose([
    #     # args.image_size + 1/4 *args.image_size
    #     # torchvision.transforms.Resize(
    #     #     args.image_size[0] + 1/4 * args.image_size[0]),
    #     # torchvision.transforms.RandomResizedCrop(
    #     #     args.image_size, scale=(0.8, 1.0)),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])

    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    (0.5,), (0.5,))
                                                 ])

    dataset = SpectrogramSet(data_path=args.dataset_path)
    # dataset = torchvision.datasets.FakeData(
    #     size=5, image_size=(1, args.image_size[0], args.image_size[1]), transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
