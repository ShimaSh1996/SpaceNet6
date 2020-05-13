from torch.utils.data import DataLoader, random_split
from core.data.datasets import SpaceNetDataset
from core.data.transforms import transforms
from copy import deepcopy

__all__ = ["SpaceNetDatalaoder"]

def SpaceNetDatalaoder(cfg):
    base_transforms = [
        transforms.ConvertToFloat(),
        transforms.Resize(
            input_shape=cfg.DATASET.IMAGE_SHAPE,
            output_shape=cfg.DATASET.IMAGE_SHAPE,
        ),
        transforms.Normalize(),
        transforms.ToTensor(),
    ]

    train_transforms = base_transforms
    # train_transforms = (
    #     base_transforms[0:1]
    #     + [
    #         transforms.PhotometricDistort(),
    #         transforms.RandomMirror(),
    #         transforms.RandomBrightness(),
    #     ]
    #     + base_transforms[1:]
    # )

    base_transforms = transforms.Compose(base_transforms)
    train_transforms = transforms.Compose(train_transforms)

    dataset = SpaceNetDataset(
        image_dir=cfg.DATASET.IMAGE_DIR,
        sar_dir=cfg.DATASET.SAR_DIR,
        csv_fp=cfg.DATASET.CSV_FILE,
        transform=base_transforms
    )

    if cfg.DATASET.IS_TRAIN:
        val_size = int(cfg.DATASET.VALIDATION_RATIO * len(dataset))
        train_size = len(dataset) - val_size

        trainset, valset = random_split(dataset, (train_size, val_size))
        trainset.dataset = deepcopy(dataset)
        trainset.dataset.transform = train_transforms
        datasets = {"train": trainset, "val": valset}
    else:
        datasets = {"test": dataset}

    data_loaders = {
        name: DataLoader(
            dataset,
            batch_size=cfg.DATA_LOADER.BATCH_SIZE,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            shuffle=True if name == "train" else False,
        )
        for name, dataset in datasets.items()
    }
    return data_loaders
