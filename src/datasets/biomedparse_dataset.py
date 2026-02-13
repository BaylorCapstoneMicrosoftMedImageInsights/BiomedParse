import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = image + noise
    return noisy_image

def add_salt_and_pepper_noise(image, noise_ratio=0.02):
    noisy_image = image.copy()
    h, w, c = noisy_image.shape
    noisy_pixels = int(h * w * noise_ratio)
    for _ in range(noisy_pixels):
        row, col = np.random.randint(0, h), np.random.randint(0, w)
        if np.random.rand() < 0.5:
            noisy_image[row, col] = [0, 0, 0] 
        else:
            noisy_image[row, col] = [255, 255, 255]
    return noisy_image

class DataAugmentation:
    def __init__(
        self,
        prob=0.5,
        rotate=True,
        flip=True,
        pixel_shift=True,
        pixel_shift_ratio=0.1,
        crop=True,
        crop_ratio=0.1,
        gaussian_noise=True,
        salt_and_pepper_noise=True,
    ):
        self.prob = prob
        self.rotate = rotate
        self.flip = flip
        self.pixel_shift = pixel_shift
        self.pixel_shift_ratio = pixel_shift_ratio
        self.crop = crop
        self.crop_ratio = crop_ratio
        self.gaussian_noise = gaussian_noise
        self.salt_and_pepper_noise = salt_and_pepper_noise

    def __call__(self, img, mask):
        if random.random() < self.prob:
            # Rotate the image
            if self.rotate:
                rotate_times = random.randint(0, 3)
                img = np.rot90(img, rotate_times, (0, 1))
                mask = np.rot90(mask, rotate_times, (0, 1))
            # Flip the image
            if self.flip:
                flip = random.choice([0, 1, -1])
                img = cv2.flip(img, flip)
                mask = cv2.flip(mask, flip)

            # Crop the image
            if self.crop and random.random() < self.prob:
                h, w = img.shape[:2]
                c = np.array([h, w]) / 2
                # pad the image with zeros
                pad = int(2 * h * self.crop_ratio + 1)
                img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
                mask_pad_val = 0
                if len(mask.shape) == 3:
                    mask = np.pad(mask, ((pad, pad), (pad, pad), (0,0)), mode="constant", constant_values=mask_pad_val)
                else:
                    mask = np.pad(mask, ((pad, pad), (pad, pad)), mode="constant", constant_values=mask_pad_val)


                # crop the image
                scale = 1 + np.random.uniform(-self.crop_ratio, self.crop_ratio)
                new_h, new_w = np.array([h, w]) * scale
                new_c = c + np.random.uniform(
                    -self.crop_ratio, self.crop_ratio, size=2
                ) * np.array([h, w])
                x1, x2 = new_c[0] - new_h / 2, new_c[0] + new_h / 2
                y1, y2 = new_c[1] - new_w / 2, new_c[1] + new_w / 2
                x1 = pad + int(x1)
                x2 = pad + int(x2)
                y1 = pad + int(y1)
                y2 = pad + int(y2)
                img = img[x1:x2, y1:y2, :]
                mask = mask[x1:x2, y1:y2]

                # resize the image
                img = cv2.resize(img, (h, w), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (h, w), interpolation=cv2.INTER_NEAREST)

            # Shift pixel values
            if self.pixel_shift:
                scale = 255.0
                shift = np.random.uniform(-scale, scale) * self.pixel_shift_ratio
                img = img + shift
                
            # Add Gaussian noise
            if self.gaussian_noise and random.random() < self.prob:
                std = np.random.uniform(0, 25)
                img = add_gaussian_noise(img, mean=0, std=std)
                
            # Add salt and pepper noise
            if self.salt_and_pepper_noise and random.random() < self.prob:
                noise_ratio = np.random.uniform(0.02, 0.1)
                img = add_salt_and_pepper_noise(img, noise_ratio=noise_ratio)

        return img, mask

def choose_prompt(prompts):
    if isinstance(prompts, str):
        return prompts
    elif isinstance(prompts, list):
        return random.choice(prompts)
    else:
        raise ValueError("Invalid prompt type. Must be str or list.")

class BiomedParseDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        num_prompts=4,
        all_class_masks=False,
        transforms=DataAugmentation(),
        img_size=(1024, 1024),
        interpolate_mask_size=(1024, 1024),
        name=None,
        negative=False,
    ):
        self.root_dir = root_dir
        self.split = split
        self.num_prompts = num_prompts
        self.all_class_masks = all_class_masks
        self.transforms = transforms
        self.img_size = img_size
        self.interpolate_mask_size = interpolate_mask_size
        self.json_file = os.path.join(root_dir, f"{split}.json")
        self.name = name if name else os.path.basename(os.path.normpath(root_dir))
        
        # This should be fetched from data.yaml, but hardcoding for now to match preprocess script
        self.CLASS_NAMES = [
            'Comminuted', 'Greenstick', 'Healthy', 'Linear', 'Oblique Displaced', 
            'Oblique', 'Segmental', 'Spiral', 'Transverse Displaced', 'Transverse'
        ]

        with open(self.json_file, "r") as file:
            self.data_info = json.load(file)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        ann_info = self.data_info[idx]
        
        img_path = ann_info.get("image")
        mask_dir = ann_info.get("mask")

        try:
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if image is None: raise IOError
        except (IOError, FileNotFoundError):
            print(f"Image not found, creating empty image: {img_path}")
            image = np.zeros((*self.img_size, 3), dtype=np.uint8)

        masks = []
        if mask_dir and os.path.isdir(mask_dir):
            for class_name in self.CLASS_NAMES:
                mask_path = os.path.join(mask_dir, f"{class_name}.png")
                try:
                    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    if m is None:
                        m = np.zeros(self.img_size, dtype=np.uint8)
                    if m.max() > 1:
                        m[m > 0] = 1 # Normalize
                    masks.append(m)
                except (IOError, FileNotFoundError):
                    print(f"Mask file not found, creating empty mask: {mask_path}")
                    masks.append(np.zeros(self.img_size, dtype=np.uint8))
        else:
            print(f"Mask directory not found or invalid, creating empty masks: {mask_dir}")
            masks = [np.zeros(self.img_size, dtype=np.uint8) for _ in range(len(self.CLASS_NAMES))]

        mask_stack = np.stack(masks, axis=-1).astype(np.uint8)

        if self.transforms:
            image, mask_stack = self.transforms(image, mask_stack)
            
        if random.random() < 0.5:
            image = image[:, :, [0, 2, 1]]
            
        image = cv2.resize(
            image,
            self.img_size,
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)

        # Ensure mask is resized correctly
        if mask_stack.shape[:2] != self.interpolate_mask_size:
             mask_stack = cv2.resize(
                mask_stack,
                self.interpolate_mask_size,
                interpolation=cv2.INTER_NEAREST,
            )

        image = np.transpose(image, (2, 0, 1))
        # Ensure mask is (C, H, W) if it's not already
        if len(mask_stack.shape) == 3 and mask_stack.shape[-1] == len(self.CLASS_NAMES):
            mask_stack = np.transpose(mask_stack, (2, 0, 1))

        return {
            "image": torch.tensor(image.copy(), dtype=torch.float32),
            "labels": torch.tensor(mask_stack.copy(), dtype=torch.long),
            "text": "multi-channel-mask",
            "class_ids": "&".join([str(i) for i in range(len(self.CLASS_NAMES))]),
            "mask_file": mask_dir,
            "instance_label": False,
            "multiclass_label": True,
        }


