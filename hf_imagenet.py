# hf_imagenet.py
#
# Simple PyTorch Dataset wrapper around
#   evanarlian/imagenet_1k_resized_256
#
# Usage:
#    from hf_imagenet import HFImageNetDataset

from typing import Callable

from datasets import load_dataset
from torch.utils.data import Dataset


class HFImageNetDataset(Dataset):
    """
    Wraps the HuggingFace dataset:
        evanarlian/imagenet_1k_resized_256

    into a standard torch.utils.data.Dataset that returns
        (image_tensor, label_int)

    Arguments
    ---------
    split: "train" or "validation"
    cache_dir: path where HF will cache the dataset (e.g. /opt/imagenet_hf)
    transform: torchvision transform taking a PIL.Image -> tensor
    """

    DATASET_NAME = "evanarlian/imagenet_1k_resized_256"

    def __init__(
        self,
        split: str = "train",
        cache_dir: str | None = None,
        transform: Callable | None = None,
    ) -> None:
        assert split in ("train", "validation"), f"Unknown split: {split}"

        # This will download (once) and then reuse local cache
        self.ds = load_dataset(
            self.DATASET_NAME,
            split=split,
            cache_dir=cache_dir,
        )

        self.transform = transform

        # HF Datasets exposes label names in the feature metadata
        label_feature = self.ds.features["label"]
        self.classes = list(label_feature.names)
        self.num_classes = len(self.classes)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        example = self.ds[idx]

        # HF returns images as PIL.Image.Image objects by default
        img = example["image"]
        label = example["label"]  # integer index into self.classes

        if self.transform is not None:
            img = self.transform(img)

        return img, label