from pathlib import Path
from torchvision.datasets import ImageFolder

def get_datasets(data_dir: str, train_transform, eval_transform):
    data_dir = Path(data_dir)
    train_ds = ImageFolder(root=str(data_dir / "train"), transform=train_transform)
    val_ds   = ImageFolder(root=str(data_dir / "val"),   transform=eval_transform)
    test_ds  = ImageFolder(root=str(data_dir / "test"),  transform=eval_transform)
    return train_ds, val_ds, test_ds