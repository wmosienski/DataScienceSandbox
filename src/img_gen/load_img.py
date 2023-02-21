import os

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as F
plt.rcParams["savefig.bbox"] = 'tight'


img_number = 3
directory = "images/train2"
size = 64
def load_imgs(steps):
    ctr = 0
    imgs = []
    ogs = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        loaded_img = (T.Resize(size)(torchvision.io.read_image(img_path)).to(torch.float32))
        ogs += [loaded_img.clone().detach()]
        imgs_with_noise = generate_noise_pairs(loaded_img, steps)
        imgs += imgs_with_noise
        if ctr >= img_number:
            break
        ctr += 1

    return imgs, ogs

def add_noise(img):
    variance = 0.002
    return (img + (variance**0.5) * torch.randn(3, size, size) * 256).clamp(0, 256)


def generate_noise_pairs(img, steps, times=4):
    pairs = []
    original = img.clone().detach()
    for t in range(times):
        prev = original.clone().detach()
        for i in range(steps):
            noised = add_noise(prev.clone().detach())
            pairs = pairs + [[noised.clone().detach(), prev.clone().detach()]]
            prev = noised

    return pairs


def show_imgs(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    for i, img in enumerate(imgs):
        # img = img.detach()
        img = F.to_pil_image(img.clamp(0, 256).to(torch.uint8))
        img.show()
