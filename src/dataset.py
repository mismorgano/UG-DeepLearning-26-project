import os
import pathlib
import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2

# ── Dataset split ─────────────────────────────────────────────────────────────
TRAIN_SIZE = 220
TEST_SIZE = 80
TRAIN_CROP = 128  # spatial size used during training
NATIVE_SIZE = 512  # full resolution used for testing

base_transform = v2.Compose(
    [
        v2.Resize((NATIVE_SIZE, NATIVE_SIZE)),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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

    def __getitem__(self, index: int):

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

        return input_patch, target_patch, img_name


from typing import List, Tuple

from .utils import list_image_files, split_filenames

Image = torch.Tensor

# ---------------------------------------------------------------------------
# Main Dataset class
# ---------------------------------------------------------------------------


class DenoisingDataset(Dataset):
    """PyTorch Dataset for paired noisy/clean image denoising.

    Supports two operating modes controlled by the *split* argument:

    ``'train'``
        Each 512×512 image pair is grid-cropped into 16 non-overlapping
        128×128 patches. Identical crops are applied to both the noisy and
        clean image to maintain pixel-level alignment. ``__len__`` returns
        the *total number of patches* across all training images.

    ``'test'``
        Full 512×512 image pairs are returned without any cropping so that
        fully-convolutional or Transformer architectures (UNet, Restormer …)
        can be evaluated on the native image resolution. ``__len__`` returns
        the number of *full images*.

    All outputs are PyTorch float tensors normalised to the range [-1, 1],
    which is the standard for Pix2Pix GAN training.

    Args:
        clean_dir:  Path to the directory that holds clean (ground-truth) images.
        noisy_dir:  Path to the directory that holds noisy images.
        noise_type: Noise type sub-folder name, e.g. ``'gaussian'``.
        split:      Dataset split — either ``'train'`` or ``'test'``.
        patch_size: Side length (pixels) of square patches. Default ``128``.
        seed:       Random seed used for deterministic train/test splitting.
                    Default ``42``.
        **kargs:    Named arguments, train_size, test_size, ratio to handle the dataset split.
                    Exactly one of them must be specified.

    Returns (per ``__getitem__`` call):
        A tuple ``(noisy_tensor, clean_tensor)`` where each tensor has shape
        ``(C, patch_size, patch_size)`` for training or ``(C, 512, 512)``
        for testing, with values in ``[-1, 1]``.
    """

    def __init__(
        self,
        clean_dir: Path,
        noisy_dir: Path,
        noise_type: str,
        split: str,
        patch_size: int = 128,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------ #
        # 1. Validate arguments                                               #
        # ------------------------------------------------------------------ #
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got '{split}'.")

        self.split = split
        self.patch_size = patch_size
        self.seed = seed
        self.noise_type = noise_type

        # ------------------------------------------------------------------ #
        # 2. Build & validate directory paths                                 #
        # ------------------------------------------------------------------ #
        self.clean_dir = clean_dir  # also know as input
        # Noisy images live in noisy/<noise_type>/
        self.noisy_dir = noisy_dir / noise_type  # also know as target

        # ------------------------------------------------------------------ #
        # 3. Gather image filenames from the clean directory,                 #
        #    sort alphabetically, then perform a deterministic split.         #
        # ------------------------------------------------------------------ #
        all_filenames = list_image_files(self.clean_dir)

        if len(all_filenames) == 0:
            raise RuntimeError(f"No valid images found in '{self.clean_dir}'.")

        train_files, test_files = split_filenames(all_filenames, seed=seed, **kwargs)

        # Keep only the subset relevant to this split
        self.filenames: List[str] = train_files if split == "train" else test_files

        # ------------------------------------------------------------------ #
        # 4. Pre-compute patch grid (training only)                           #
        # ------------------------------------------------------------------ #
        # For training we turn every image into a flat list of (img_idx, row, col)
        # tuples so __getitem__ can address each patch with a single integer index.
        # avoids calculate the (img_idx, row, col) on the fly at expeses of memory
        if split == "train":
            patches_per_dim = NATIVE_SIZE // patch_size  # e.g. 512//128 = 4
            self.patches_per_image = patches_per_dim * patches_per_dim  # e.g. 16
            self._patch_index: List[Tuple[int, int, int]] = [
                (img_idx, row, col)
                for img_idx in range(len(self.filenames))
                for row in range(patches_per_dim)
                for col in range(patches_per_dim)
            ]
        else:
            self.patches_per_image = 1  # not used, but kept for clarity
            self._patch_index = []

        # ------------------------------------------------------------------ #
        # 5. Shared transform: PIL Image → float tensor in [-1, 1]           #
        # ------------------------------------------------------------------ #
        # ToTensor() converts uint8 PIL [0, 255] → float32 [0.0, 1.0].
        # The Normalize step then maps [0, 1] → [-1, 1] via (x - 0.5) / 0.5.
        self._to_tensor = v2.Compose(
            [
                v2.ToDtype(dtype=torch.float32, scale=True),  # [0, 1]
                v2.Normalize(
                    [0.5, 0.5, 0.5],  # → [-1, 1]
                    [0.5, 0.5, 0.5],
                ),
            ]
        )

    # ---------------------------------------------------------------------- #
    # Length                                                                   #
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        """Return number of *patches* (train) or *full images* (test)."""
        if self.split == "train":
            return len(self._patch_index)
        return len(self.filenames)

    # ---------------------------------------------------------------------- #
    # Loading helpers                                                          #
    # ---------------------------------------------------------------------- #

    def _load_image_pair(self, filename: str) -> Tuple[Image, Image]:
        """Load a single (noisy, clean) image pair as uint8 tensor.

        Args:
            filename: Bare filename shared by both the clean and noisy dirs.

        Returns:
            ``(noisy_pil, clean_pil)`` — both are decoded uint8 tensors.

        Raises:
            FileNotFoundError: If either image file is missing.
        """
        clean_path = self.clean_dir / filename
        noisy_path = self.noisy_dir / filename

        if not clean_path.exists():
            raise FileNotFoundError(f"Clean image not found: {clean_path}")
        if not noisy_path.exists():
            raise FileNotFoundError(f"Noisy image not found: {noisy_path}")

        # Convert to RGB to handle grayscale / RGBA sources uniformly
        clean_img = decode_image(clean_path)
        noisy_img = decode_image(noisy_path)

        return noisy_img, clean_img

    def _crop_patch(
        self,
        noisy_img: Image,
        clean_img: Image,
        row: int,
        col: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract the patch at grid position (row, col) from an image pair.

        The *same* bounding box is applied to both images so noisy and clean
        patches remain spatially aligned.

        Args:
            noisy_img: Full-size uint8 tensor (noisy).
            clean_img: Full-size uint8 tensor (clean ground truth).
            row:       Row index of the patch in the grid (0-based).
            col:       Column index of the patch in the grid (0-based).

        Returns:
            ``(noisy_patch_tensor, clean_patch_tensor)`` — each with shape
            ``(C, patch_size, patch_size)`` and values in ``[-1, 1]``.
        """
        top = row * self.patch_size
        left = col * self.patch_size

        # TF.crop(img, top, left, height, width) — identical crop for both
        noisy_patch = TF.crop(noisy_img, top, left, self.patch_size, self.patch_size)
        clean_patch = TF.crop(clean_img, top, left, self.patch_size, self.patch_size)

        return noisy_patch, clean_patch

    # ---------------------------------------------------------------------- #
    # __getitem__                                                              #
    # ---------------------------------------------------------------------- #

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a single sample (patch or full image pair) by index.

        Args:
            index: Integer index in ``[0, len(dataset))``.

        Returns:
            ``(noisy_tensor, clean_tensor)`` — shapes:

            * Training:  ``(3, patch_size, patch_size)`` in ``[-1, 1]``
            * Test:      ``(3, 512, 512)``               in ``[-1, 1]``
        """
        if self.split == "train":
            # ----------------------------------------------------------------
            # Training path — extract a specific grid patch from the image
            # ----------------------------------------------------------------
            img_idx, row, col = self._patch_index[index]
            filename = self.filenames[img_idx]

            noisy_img, clean_img = self._load_image_pair(filename)
            noisy_out, clean_out = self._crop_patch(noisy_img, clean_img, row, col)

        else:
            # ----------------------------------------------------------------
            # Test path — return the full 512×512 image pair without cropping
            # ----------------------------------------------------------------
            filename = self.filenames[index]
            noisy_img, clean_img = self._load_image_pair(filename)

            noisy_out = self._to_tensor(noisy_img)
            clean_out = self._to_tensor(clean_img)

        noisy_tensor = self._to_tensor(noisy_out)
        clean_tensor = self._to_tensor(clean_out)

        return noisy_tensor, clean_tensor

    # ---------------------------------------------------------------------- #
    # Informational repr                                                       #
    # ---------------------------------------------------------------------- #

    def __repr__(self) -> str:
        return (
            f"DenoisingDataset("
            f"split='{self.split}', "
            f"noise_type='{self.noise_type}', "
            f"num_images={len(self.filenames)}, "
            f"total_samples={len(self)}, "
            f"patch_size={self.patch_size}, "
            f"seed={self.seed})"
        )
