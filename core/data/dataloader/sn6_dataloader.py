from torch.utils.data import DataLoader, random_split
from core.data.dataset import SpaceNetDataset
from core.data.transforms import transforms
from copy import deepcopy


def SpaceNetDatalaoder(cfg):
    dataset = SpaceNetDataset(
        image_dir=cfg.DATASET.IMAGE_DIR,
        sar_dir=cfg.DATASET.SAR_DIR,
        csv_dir=cfg.DATASET.CSV_DIR,
    )

    if cfg.DATASET.IS_TRAIN:
        val_size = int(cfg.DATASET.VALIDATION_RATIO * len(dataset))
        train_size = len(dataset) - val_size

        trainset, valset = random_split(dataset, (train_size, val_size))
        # trainset.dataset = deepcopy(dataset)
        # trainset.dataset.transform = train_transforms
        datasets = {"train": trainset, "val": valset}
    else:
        datasets = {"test": dataset}

    data_loaders = {
        name: DataLoader(
            dataset,
            batch_size=cfg.DATA_LOADER.BATCH_SIZE,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            shuffle=True,
        )
        for name, dataset in datasets.items()
    }
    return data_loaders
