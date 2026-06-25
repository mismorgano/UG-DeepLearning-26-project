import random
from pathlib import Path
from typing import List, Tuple

# actually dataset only contains .jpg files
VALID_IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tiff", ".bmp")


def list_image_files(directory: Path) -> List[str]:
    """Return a sorted list of image filenames found directly inside *directory*.

    Only files whose extension appears in VALID_IMAGE_EXTENSIONS are included.
    Hidden files (names starting with '.') are silently ignored.

    Args:
        directory: Path object pointing to the image folder.

    Returns:
        Alphabetically sorted list of bare filenames (not full paths).

    Raises:
        FileNotFoundError: If *directory* does not exist.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    filenames = [
        f.name
        for f in directory.iterdir()
        if f.is_file()
        and not f.name.startswith(".")  # avoid possible hidden files
        and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]

    # Alphabetical sort guarantees consistent ordering across all OS platforms
    # (e.g. macOS/Linux differ in how glob returns entries by default).
    filenames.sort()
    return filenames


def split_filenames(
    filenames: List[str],
    seed: int,
    train_size: int | None = None,
    test_size: int | None = None,
    ratio: float | None = None,
) -> Tuple[List[str], List[str]]:
    """Shuffle *filenames* deterministically and split into train / test subsets.

    A dedicated random.Random instance (seeded with *seed*) is used so this
    function never disturbs the global random state of the caller's program.

    Args:
        filenames: Alphabetically sorted list of image filenames.
        seed:      Integer seed that controls the shuffle.
        train_size:Number of images that should be used for training
        test_size: Number of images that should be used for testing/evaluationg
        ratio:     Fraction of images assigned to the training set.

    Only one of train_size, test_size, ratio is required.

    Returns:
        (train_filenames, test_filenames) — two non-overlapping lists.
    """
    # Use a local RNG so we never pollute the caller's global random state.
    rng = random.Random(seed)

    # shuffle (image) filenames
    shuffled = filenames.copy()
    rng.shuffle(shuffled)

    # check how many "size" parameters were provided
    only_one_argument_required(train_size=train_size, test_size=test_size, ratio=ratio)

    dataset_size = len(filenames)

    if train_size is not None:
        assert train_size < dataset_size
        split_index = train_size
    elif test_size is not None:
        assert test_size < dataset_size
        split_index = test_size
    else:
        split_index = int(len(shuffled) * ratio)

    train_files = shuffled[:split_index]
    test_files = shuffled[split_index:]

    return train_files, test_files


def only_one_argument_required(**kargs):
    count = sum(1 for arg in kargs.values() if arg is not None)
    if count != 1:
        raise TypeError(
            f'You must provide exactly one "size" argument. \nYou provided: {count}.'
        )
    for key, value in kargs.items():
        if value is not None:
            return key, value
