import gc
import logging
import warnings
import yaml
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
import torchmetrics as tm
from omegaconf import DictConfig, ListConfig, OmegaConf
from rich import progress
from torch.utils.tensorboard import SummaryWriter
from tqdm import TqdmExperimentalWarning

from src.dataset.dataset import DisprotDataset
from src.dataset.utils import pad_collate
from src.models import FNN, CNN
from src.utils import *

@hydra.main(version_base=None, config_path="../../config", config_name="finetune")
def main(config: DictConfig):
    # Set up logging
    logger = setup_logger()
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

    # Parse config
    logger.info("Parsing parameters")
    embedding_type = config.embedding_type
    embedding_dim = embedding_dimensions[embedding_type]

    dataset = config.dataset.name
    embedding_file = config.dataset.embedding_file

    score_type = config.score_type
    if isinstance(score_type, ListConfig):
        disprot_score_files = [Path(config.dataset.data_directory) / f"{type}.fasta" for type in score_type]
        multiple_score_types = True
    else:
        disprot_score_files = [Path(config.dataset.data_directory) / f"{score_type}.fasta"]
        multiple_score_types = False

    train_batch_size = config.training.batch_size
    learning_rate = config.training.learning_rate
    max_epochs = config.training.max_epochs
    load_full_dataset = "load_full_dataset" in config.training

    # Set up artifacts & directories
    score_type_name = "multiple" if isinstance(score_type, ListConfig) else score_type
    run_name = f"{datetime.now():%m-%d_%H:%M}_{dataset}_{score_type_name}_{embedding_type}_finetune"

    project_root = Path.cwd()
    artifact_dir = project_root / "models"
    model_dir = artifact_dir / config.run_name

    # PyTorch boilerplate
    logger.info("Setting up Torch")
    if torch.cuda.is_available():
        model_device = "cuda"
        data_device = "cuda" if load_full_dataset else "cpu"
        default_dtype = torch.float16
        use_amp = True
        map_location = None
    else:
        model_device, data_device = "cpu", "cpu"
        default_dtype = torch.bfloat16
        use_amp = False
        map_location = torch.device("cpu")

    logger.info(
        f"Using device {model_device} and automatic mixed precision with default dtype {default_dtype}"
    )

    logger.info("Loading training data")
    train_ds = torch.utils.data.ConcatDataset([
        DisprotDataset(
            embedding_file,
            score_file,
            device=data_device
        )
        for score_file in disprot_score_files
    ])

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_batch_size,
        collate_fn=pad_collate
    )

    logger.info("Setting up model")

    model = load_model(model_dir / "config.yml", model_dir / "final.pt", map_location).to(model_device)
    model = freeze_layers(model, config.freeze_layers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    model.train()

    criterion = torch.nn.BCELoss()
    auc = tm.classification.BinaryAUROC()
    metric_names = ["loss", "auc"]
    metrics = {metric_name: tm.aggregation.MeanMetric().to(model_device) for metric_name in metric_names}
    writer = SummaryWriter(log_dir="runs/" + run_name)

    logger.info("Starting training")
    with progress.Progress(
        *progress.Progress.get_default_columns(), progress.TimeElapsedColumn()
    ) as pbar:
        overall_progress = pbar.add_task("Overall", total=len(train_dl))
        for epoch in range(max_epochs):
            for metric_name in metric_names:
                metrics[metric_name].reset()

            for embs, scores, mask in train_dl:
                model.zero_grad()
                if not load_full_dataset:
                    embs = embs.to(model_device)
                    mask = mask.to(model_device)
                    scores = scores.to(model_device)
                pred = model(embs).masked_select(mask)
                scores = scores.masked_select(mask).to(dtype=torch.float32)
                loss = criterion(pred, scores)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                metrics["loss"].update(
                    loss.detach(), embs.shape[0]
                )
                metrics["auc"].update(
                    auc(pred, scores), embs.shape[0]
                )

            train_loss = metrics["loss"].compute().item()
            writer.add_scalar("loss/train", train_loss, epoch+1)
            writer.add_scalar("auc/train", metrics["auc"].compute().item(), epoch+1)
            scheduler.step(train_loss)
            pbar.advance(overall_progress)

    model_path = model_dir / f"finetuned_{dataset}_{score_type_name}.pt"
    logging.info(f"Saving model to {model_path}")
    torch.save(model.state_dict(), model_path)

    config_path = model_path.parent / (model_path.stem + ".yml")
    with open(config_path, "w+") as f:
        OmegaConf.save(config=config, f=f)

if __name__ == "__main__":
    main()
