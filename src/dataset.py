import os
import pathlib
import random
from pathlib import Path

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image

# ── Dataset split ─────────────────────────────────────────────────────────────
TRAIN_SIZE = 220
TEST_SIZE = 80
TRAIN_CROP = 128  # spatial size used during training
NATIVE_SIZE = 512  # full resolution used for testing

base_transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


class Pix2PixPatchDataset(Dataset):
    """
    Paired (noisy, clean) dataset.

    Parameters
    ----------
    image_paths : list[Path]
        Paths to the *clean* source images.
    mode : str
        'train'  → resize shorter side to NATIVE_SIZE then random-crop to
                   TRAIN_CROP.  Horizontal flip augmentation applied.
        'test'   → resize to NATIVE_SIZE × NATIVE_SIZE (no crop).
    degradation : NoiseDegradation
        Instance that produces a noisy version of a clean tensor on-the-fly.
    """

    def __init__(
        self,
        input_dir: Path,
        target_dir: Path,
        patch_size=TRAIN_CROP,
        transform=None,
        mode="train",
    ):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.patch_size = patch_size
        self.mode = mode

        # the filenames match exactly in both directories
        self.image_filenames = sorted(os.listdir(input_dir))

        self.patches_per_side = NATIVE_SIZE // patch_size
        self.patches_per_image = self.patches_per_side * self.patches_per_side

        # base transformations (Scaling to 512, just ot be sure)
        # self.base_transform = transforms.Compose(
        #     [
        #         transforms.Resize((512, 512)),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        #     ]
        # )

    def __len__(self):
        return len(self.image_filenames) * self.patches_per_image

    def __getitem__(self, index):

        # determine which image this index belongs to
        img_idx = index // self.patches_per_image
        # determine which of the 16 patches we are extracting from that image
        patch_idx = index % self.patches_per_image

        # load the paired images
        img_name = self.image_filenames[img_idx]
        input_img_path = self.input_dir / img_name
        input_img = decode_image(input_img_path)
        target_img_path = self.target_dir / img_name
        target_img = decode_image(target_img_path)

        # apply base transformations (coerce them to 512x512 Tensors)
        if self.transform:
            input_tensor = self.transform(input_img)
            target_tensor = self.transform(target_img)

        # calculate grid row and column coordinates
        row = patch_idx // self.patches_per_side
        col = patch_idx % self.patches_per_side

        # transfer row/col grid coordinates to pixel indices
        top = row * self.patch_size
        left = col * self.patch_size

        # extract the exact non-overlaping crop
        input_patch = TF.crop(input_tensor, top, left, self.patch_size, self.patch_size)
        target_patch = TF.crop(
            target_tensor, top, left, self.patch_size, self.patch_size
        )

        return input_patch, target_patch
