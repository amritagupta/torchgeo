from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import yaml
from matplotlib.figure import Figure
from torch.utils.data import Dataset

from torchgeo.datasets import NonGeoDataset
from torchgeo.datasets.errors import DatasetNotFoundError
from torchgeo.datasets.utils import check_integrity, download_url, extract_archive


class KakumaAerialBase(Dataset[dict[str, Any]]):
    """Kakuma Aerial Dataset (Version 1.0.1).

    The `Kakuma Aerial <https://zenodo.org/record/14727217>`__ dataset consists of high-resolution
    aerial imagery of the Kakuma and Kalobeyei refugee camps in Turkana County, Kenya. The dataset
    supports a variety of tasks, including:

    * Building and solar panel segmentation
    * Roof material classification
    * Toilet identification

    If you use this dataset in your research, please cite:

    A. Gupta, A. Ortiz, S. F. Nsutezo, D. Kebut, S. Iyer, R. Dodhia, and J. M. Lavista Ferres,
    "Mapping Refugee Camps with AI: A Benchmark Dataset and Baseline Models for Humanitarian Applications,"
    in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops,
    Tucson, AZ, USA, Mar. 2025.

    The dataset is released under the `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`__ license.

    .. versionadded:: 1.0.1
    """

    metadata = {
        "base_url": "https://zenodo.org/records/14727217/files/",
        "files": {
            "tiled_dataset_splits.yml": "67bfb9e2051bf684444749096120c51b",
            "chipped_segmentation-background-building-solar.zip": "83194297d95b93b43fb052c8e45f1ad7",
            "chipped_segmentation-background-building_boundary-building-solar.zip": "6f494110cdc316fdf20aed6c0d72294c",
            "chipped_segmentation-background-building-building_boundary-solar.zip": "6b38dfd9fc1e6feaa164ecfb7e0ec3fb",
            "chipped_roof_material_classification.zip": "278492cd03bf7feb3e816a8efc5bc26a",
            "chipped_toilet_classification.zip": "0dab69a5899736505f79a6b5373367c9",
        },
        "version": "1.0.1",
    }

    def __init__(
        self,
        root: str | Path = "data",
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Kakuma Aerial dataset instance.
        Args:
            root: root directory where the dataset is stored
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 checksum of the downloaded files

        Raises:
            DatasetNotFoundError: if the dataset is not found or corrupted and download is False
        """
        self.root = Path(root)
        self.download = download
        self.checksum = checksum

        # Verify and load the tiled dataset splits
        self._verify("tiled_dataset_splits.yml")
        with open(self.root / "tiled_dataset_splits.yml", "r") as f:
            self.splits = yaml.safe_load(f)

    def _verify(self, file: str = "tiled_dataset_splits.yml") -> None:
        """Verify the presence and integrity of the specified file."""
        md5 = self.metadata["files"][file]
        file_path = self.root / file

        if not file_path.exists() or (
            self.checksum and not check_integrity(file_path, md5)
        ):
            if self.download:
                if file_path.exists():
                    print(f"File '{file_path}' failed verification. Redownloading...")
                    file_path.unlink()  # Remove corrupted file
                else:
                    print(f"File '{file_path}' not found. Downloading...")
                self._download(file)
            else:
                error_type = "corrupted" if file_path.exists() else "missing"
                raise DatasetNotFoundError(
                    f"File '{file_path}' is {error_type}. "
                    f"Set `download=True` to {'redownload' if error_type == 'corrupted' else 'download'} it."
                )
        # If the file is a zip archive and has not been extracted, extract it
        if file_path.suffix == ".zip" and not (self.root / file_path.stem).exists():
            extract_archive(str(file_path))

    def _download(self, file: str = "tiled_dataset_splits.yml") -> None:
        """Download the specified file.

        Args:
            file_key: the key of the file in the `metadata['files']` dictionary
        """
        md5 = self.metadata["files"][file]
        url = f"{self.metadata['base_url']}{file}?download=1"

        # Create the root directory if it doesn't exist
        self.root.mkdir(parents=True, exist_ok=True)

        # Download the file
        download_url(
            url, str(self.root), filename=file, md5=md5 if self.checksum else None
        )


class KakumaAerialSegmentation(KakumaAerialBase, NonGeoDataset):
    """The `Kakuma Aerial <https://zenodo.org/record/14727217>`__ dataset consists of high-resolution
    aerial imagery of the Kakuma and Kalobeyei refugee camps in Turkana County, Kenya. This class
    supports building and solar panel segmentation on a pre-chipped dataset.

    If you use this dataset in your research, please cite:

    A. Gupta, A. Ortiz, S. F. Nsutezo, D. Kebut, S. Iyer, R. Dodhia, and J. M. Lavista Ferres,
    "Mapping Refugee Camps with AI: A Benchmark Dataset and Baseline Models for Humanitarian Applications,"
    in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops,
    Tucson, AZ, USA, Mar. 2025.

    The dataset is released under the `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`__ license.

    .. versionadded:: 1.0.1
    """

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        mask_type: str = "background-building-solar",
        transforms: Optional[Callable] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Kakuma Aerial segmentation dataset instance.

        Args:
            root: root directory where the dataset is stored
            split: one of 'train', 'val', or 'test'
            mask_type: type of mask to load, one of 'background-building-solar',
                'background-building_boundary-building-solar', or 'background-building-building_boundary-solar'
            transforms: a function/transform that takes in a sample and returns a transformed version
            download: if True, download the dataset and store it in the root directory
            checksum: if True, check the MD5 checksum of the downloaded files
        """
        # Call parent class constructor to verify the tiled dataset splits file
        super().__init__(root=root, download=download, checksum=checksum)

        # Check the split and mask type arguments
        assert split in [
            "train",
            "val",
            "test",
        ], f"Invalid split '{split}', must be 'train', 'val', or 'test'"
        if mask_type not in [
            "background-building-solar",
            "background-building_boundary-building-solar",
            "background-building-building_boundary-solar",
        ]:
            raise ValueError(
                f"Invalid mask type '{mask_type}'. Available types: "
                "'background-building-solar', 'background-building_boundary-building-solar', "
                "'background-building-building_boundary-solar'"
            )
        self.split = split
        self.mask_type = mask_type
        self.transforms = transforms

        # Verify the chipped dataset file
        self.file_name = "chipped_segmentation-" + mask_type + ".zip"
        self._verify(self.file_name)

        # Load the image and mask files for the current split
        self._load_files()

    def _load_files(self):
        """Load the image and mask files for the current split."""
        # Chips were downloaded to a directory with the same name as the zip file
        dataset_dir = self.root / self.file_name.replace(".zip", "")
        images_dir = dataset_dir / "images"
        image_files = list(images_dir.rglob("*.tif"))
        masks_dir = dataset_dir / "masks"
        mask_files = list(masks_dir.rglob("*.tif"))
        split_tiles = self.splits[self.split + "_tiles"]
        split_image_files = [
            f
            for f in image_files
            if any(tile + "_" in str(f.stem) for tile in split_tiles)
        ]
        split_mask_files = [
            f
            for f in mask_files
            if any(tile + "_" in str(f.stem) for tile in split_tiles)
        ]
        # Make sure the stems match
        assert set([f.stem for f in split_image_files]) == set(
            [f.stem for f in split_mask_files]
        ), "Mismatch between image and mask files"

        # Order the files by stem
        split_image_files = sorted(split_image_files, key=lambda f: f.stem)
        split_mask_files = sorted(split_mask_files, key=lambda f: f.stem)

        # Return a list of paired image-mask paths
        self.chip_paths = [
            {"image": str(img), "mask": str(msk)}
            for img, msk in zip(split_image_files, split_mask_files)
        ]

    def _load_image(self, index: int) -> torch.Tensor:
        """Load the image data for the chip at the given index.

        Args:
            index: index of the chip to load.

        Returns:
            torch.Tensor: A tensor containing the image data.
        """
        chip_path = self.chip_paths[index]["image"]
        with rasterio.open(chip_path) as image_data:
            image_array = image_data.read()[0:3, :, :].astype(float)
        image_tensor = torch.from_numpy(image_array)
        return image_tensor

    def _load_target(self, index):
        """Load the target data for the chip at the given index.

        Args:
            index: index of the chip to load.

        Returns:
            torch.Tensor: A tensor containing the target data.
        """
        mask_path = self.chip_paths[index]["mask"]
        with rasterio.open(mask_path) as target_data:
            target_array = target_data.read(1).astype(np.int32)
        target_tensor = torch.from_numpy(target_array)
        return target_tensor

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Retrieve the image and mask data for the chip at the given index."""
        image = self._load_image(index)
        mask = self._load_target(index)

        sample = {"image": image, "mask": mask}

        if self.transforms:
            sample = self.transforms(sample)

        # Replace nodata with background
        sample["mask"][sample["mask"] == 0] = 1

        return sample

    def plot(self, sample, show_titles=False, **kwargs):
        """
        Plot the image, mask, and optionally the prediction from a sample.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            **kwargs: arbitrary keyword arguments for the imshow function

        Returns:
            a matplotlib Figure with the rendered sample
        """

        def plot_image(ax, image, title=None):
            ax.imshow(image, **kwargs)
            ax.axis("off")
            if title and show_titles:
                ax.set_title(title)

        if "prediction" in sample:
            prediction = sample["prediction"]
            n_cols = 3
        else:
            n_cols = 2
        image, mask = sample["image"], sample["mask"]

        fig, axs = plt.subplots(nrows=1, ncols=n_cols, figsize=(10, n_cols * 5))
        plot_image(axs[0], image.permute(1, 2, 0), "Image")
        plot_image(axs[1], mask, "Mask")

        if "prediction" in sample:
            plot_image(axs[2], prediction, "Prediction")

        return fig

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.chip_paths)


class KakumaAerialRoofClassification(KakumaAerialBase, NonGeoDataset):
    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        # Call parent class constructor to verify the tiled dataset splits file
        super().__init__(root=root, download=download, checksum=checksum)

        # Check the split argument
        assert split in ["train", "val", "test"], f"Invalid split '{split}'"
        self.split = split
        self.transforms = transforms
        self.file_name = "chipped_roof_material_classification.zip"

        # Verify the chipped dataset file
        self._verify(self.file_name)

        # Load the chip files for the current split
        self._load_files()

    def _load_files(self):
        # Chips were downloaded to a directory with the same name as the zip file
        dataset_dir = self.root / self.file_name.replace(".zip", "")
        label_to_class = {i: c.name for i, c in enumerate(dataset_dir.iterdir())}
        class_to_label = {c: i for i, c in label_to_class.items()}
        split_tiles = self.splits[self.split + "_tiles"]
        split_chip_files = [
            [dataset_dir / c / f.name, l]
            for c, l in class_to_label.items()
            for f in (dataset_dir / c).iterdir()
            if any(tile + "_" in f.stem for tile in split_tiles)
        ]
        # Sort the files by stem
        split_chip_files = sorted(split_chip_files, key=lambda f: f[0].stem)

        # Return a list of paired chip paths and labels
        self.chip_paths = [
            {"image": str(chip), "label": label} for chip, label in split_chip_files
        ]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return a single sample."""
        chip_path = self.chip_paths[index]["image"]
        label = self.chip_paths[index]["label"]

        with rasterio.open(chip_path) as chip_data:
            chip_array = chip_data.read().astype(float)
        chip_tensor = torch.from_numpy(chip_array)

        sample = {"image": chip_tensor, "label": label}

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.chip_paths)

    def plot(self, sample: dict[str, torch.Tensor], show_mask: bool = False) -> Figure:
        sample_image, sample_label = sample["image"], sample["label"]
        sample_img = sample_image[0:3, :, :].numpy()
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(sample_img.transpose(1, 2, 0))
        if show_mask:
            sample_mask = sample_image[3, :, :].numpy()
            ax.imshow(sample_mask, alpha=0.5)
        ax.set_title(f"Label: {sample_label}")
        ax.axis("off")
        return fig


class KakumaAerialToiletClassification(KakumaAerialBase, NonGeoDataset):
    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        # Call parent class constructor to verify the tiled dataset splits file
        super().__init__(root=root, download=download, checksum=checksum)

        # Check the split argument
        assert split in ["train", "val", "test"], f"Invalid split '{split}'"
        self.split = split
        self.transforms = transforms
        self.file_name = "chipped_toilet_classification.zip"

        # Verify only the chipped dataset file
        self._verify(self.file_name)

        # Load the chip files for the current split
        self._load_files()

    def _load_files(self):
        # Chips were downloaded to a directory with the same name as the zip file
        dataset_dir = self.root / self.file_name.replace(".zip", "")
        label_to_class = {i: c.name for i, c in enumerate(dataset_dir.iterdir())}
        class_to_label = {c: i for i, c in label_to_class.items()}
        split_tiles = self.splits[self.split + "_tiles"]
        split_chip_files = [
            [dataset_dir / c / f.name, l]
            for c, l in class_to_label.items()
            for f in (dataset_dir / c).iterdir()
            if any(tile + "_" in f.stem for tile in split_tiles)
        ]
        # Sort the files by stem
        split_chip_files = sorted(split_chip_files, key=lambda f: f[0].stem)

        # Return a list of paired chip paths and labels
        self.chip_paths = [
            {"image": str(chip), "label": label} for chip, label in split_chip_files
        ]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return a single sample."""
        chip_path = self.chip_paths[index]["image"]
        label = self.chip_paths[index]["label"]

        with rasterio.open(chip_path) as chip_data:
            chip_array = chip_data.read().astype(float)
        chip_tensor = torch.from_numpy(chip_array)

        sample = {"image": chip_tensor, "label": label}

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.chip_paths)

    def plot(self, sample: dict[str, torch.Tensor], show_mask: bool = True) -> Figure:
        sample_image, sample_label = sample["image"], sample["label"]
        sample_img = sample_image[0:3, :, :].numpy()
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(sample_img.transpose(1, 2, 0))
        if show_mask:
            sample_mask = sample_image[3, :, :].numpy()
            ax.imshow(sample_mask, alpha=0.5)
        ax.set_title(f"Label: {sample_label}")
        ax.axis("off")

        return fig
