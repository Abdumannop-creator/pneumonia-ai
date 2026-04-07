import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from ml.preprocessing.transforms import get_transforms
from ml.datasets.pneumonia_dataset import get_datasets
from ml.models.model_factory import create_model
from ml.training.train import train_one_epoch
from ml.training.evaluate import evaluate


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    # ✅ Always resolve paths from project root
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    cfg_path = PROJECT_ROOT / "ml" / "config" / "train_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    set_seed(cfg["seed"])

    data_dir = PROJECT_ROOT / cfg["data_dir"]
    img_size = cfg["img_size"]
    batch_size = cfg["batch_size"]
    num_workers = cfg["num_workers"]

    # Windows/CPU stability tip
    if num_workers is None:
        num_workers = 0

    train_tf, eval_tf = get_transforms(img_size)
    train_ds, val_ds, test_ds = get_datasets(str(data_dir), train_tf, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = "cpu"
    print("Device:", device)
    print("Classes:", train_ds.classes, "class_to_idx:", train_ds.class_to_idx)

    out_dir = PROJECT_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    best = {"model": None, "val_auc": -1.0, "state_path": None, "val_metrics": None}

    for model_name in cfg["model_candidates"]:
        print(f"\n=== Training: {model_name} ===")
        model = create_model(model_name, num_classes=2).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, cfg["epochs"] + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, device)
            print(
                f"epoch {epoch}/{cfg['epochs']} "
                f"loss={loss:.4f} "
                f"val_auc={val_metrics['auc']:.4f} "
                f"val_f1={val_metrics['f1']:.4f} "
                f"val_recall={val_metrics['recall']:.4f}"
            )

        # save checkpoint for this model
        ckpt_path = out_dir / f"{model_name}.pt"
        torch.save(model.state_dict(), ckpt_path)

        # evaluate on val for selection
        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["auc"] > best["val_auc"]:
            best.update({
                "model": model_name,
                "val_auc": float(val_metrics["auc"]),
                "state_path": ckpt_path,          # ✅ keep as Path
                "val_metrics": val_metrics
            })

    print("\n=== Best Model Selected (by VAL AUC) ===")
    print({"model": best["model"], "val_auc": best["val_auc"], "val_metrics": best["val_metrics"]})

    if best["model"] is None or best["state_path"] is None:
        raise RuntimeError("Best model selection failed. Check dataset and training logs.")

    # load best, evaluate on test, and export as final
    best_model = create_model(best["model"], num_classes=2).to(device)
    best_model.load_state_dict(torch.load(best["state_path"], map_location=device))

    test_metrics = evaluate(best_model, test_loader, device)
    print("Test metrics:", test_metrics)

    best_model_path = PROJECT_ROOT / cfg["best_model_path"]
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_model.state_dict(), best_model_path)

    meta = {
        "model_name": best["model"],
        "img_size": img_size,
        "classes": train_ds.classes,
        "class_to_idx": train_ds.class_to_idx,
        "selection": {
            "val_metrics": best["val_metrics"],
            "val_auc": best["val_auc"]
        },
        "test_metrics": test_metrics,
        "threshold": 0.5
    }

    meta_path = PROJECT_ROOT / cfg["meta_path"]
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nSaved artifacts:")
    print(" -", str(best_model_path))
    print(" -", str(meta_path))


if __name__ == "__main__":
    main()