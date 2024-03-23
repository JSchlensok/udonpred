import gc
import logging
import json
import gc
import logging
import json
import warnings
from datetime import datetime
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import hydra
import numpy as np
import polars as pl
import torch
import torchmetrics as tm
from Bio import SeqIO
from maskedtensor import masked_tensor
from omegaconf import DictConfig, OmegaConf
from omegaconf import DictConfig, OmegaConf
from rich import progress
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

from src.dataset.clustering.parse import read_cluster_assignments
from src.dataset.dataset import TriZodDataset
from src.dataset.utils import ClusterSampler, pad_collate
from src.models import FNN, CNN
from src.utils.logging import setup_logger
from src.utils import embedding_dimensions, model_classes

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig):
    logger = setup_logger()
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

    logger.info("Parsing parameters")
    embedding_type = config.embedding_type
    embedding_dim = embedding_dimensions[embedding_type]

    dataset = config.dataset.name
    score_type = config.score_type
    embedding_file = config.dataset.embedding_file
    cluster_assignments_file = config.dataset.cluster_file
    trizod_score_file = config.dataset.trizod_file

    train_batch_size = config.training.batch_size
    learning_rate = config.training.learning_rate
    n_splits = config.training.n_splits
    max_epochs = config.training.max_epochs

    final_model = (n_splits == 1)

    val_batch_size = None if final_model else config.validation.batch_size

    model_class = model_classes[config.model.type]
    model_config = config.model.params

    run_name = f"{datetime.now():%m-%d_%H:%M}_{dataset}_{embedding_type}_{config.model.type}_{score_type}"
    project_root = Path.cwd()
    artifact_dir = project_root / "models"
    model_dir = artifact_dir / f"{run_name}_epoch_{max_epochs}"
    model_dir.mkdir(exist_ok=True, parents=True)
    artifact_dir = project_root / "models"
    model_dir = artifact_dir / f"{run_name}_epoch_{max_epochs}"
    model_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Setting up Torch")
    if torch.cuda.is_available():
        device = "cuda"
        default_dtype = torch.float16
        use_amp = True
    else:
        device = "cpu"
        default_dtype = torch.bfloat16
        use_amp = False

    logger.info(
        f"Using device {device} and automatic mixed precision withh default dtype {default_dtype}"
    )

    torch.set_default_device(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # TODO add option for training only one model
    logger.info("Initializing dataset")
    all_cluster_assignments = read_cluster_assignments(
      cluster_assignments_file
    )
    sequence_ids = np.array(
        all_cluster_assignments.select("sequence_id").to_series(), dtype=str
    )

    if not final_model:
        kfold = KFold(n_splits=n_splits, shuffle=False)
        splits = list(kfold.split(sequence_ids))

        criterion = torch.nn.MSELoss()

        # TODO add option for more metrics
        split_names = ["train", "val"]
        metric_names = ["loss"]
        metrics = {split: {metric_name: tm.aggregation.MeanMetric() for metric_name in metric_names} for split in split_names}
        foldwise_metric_values = {
            split_name: {
                metric_name: {
                    fold: [] for fold in range(n_splits)
                }
                for metric_name in metric_names
            }
            for split_name in split_names
        }

        with progress.Progress(
            *progress.Progress.get_default_columns(), progress.TimeElapsedColumn()
        ) as pbar:
            overall_progress = pbar.add_task("Overall", total=n_splits)
            for fold in range(n_splits):
                train_indices, val_indices = splits[fold]
                logger.info("Loading training data")
                train_ds = TriZodDataset(
                    embedding_file,
                    trizod_score_file,
                    sequence_ids[train_indices],
                    all_cluster_assignments.filter(
                        pl.col("cluster_representative_id").is_in(sequence_ids[train_indices])
                    ),
                    device=device,
                    score_type=score_type
                )
                train_dl = torch.utils.data.DataLoader(
                    train_ds,
                    batch_size=train_batch_size,
                    sampler=ClusterSampler(train_ds.clusters, shuffle=True),
                    collate_fn=pad_collate
                )
                logger.info("Loading validation data")
                val_ds = TriZodDataset(
                    embedding_file,
                    trizod_score_file,
                    sequence_ids[val_indices],
                    all_cluster_assignments.filter(
                        pl.col("cluster_representative_id").is_in(sequence_ids[val_indices])
                    ),
                    device=device
                )
                val_dl = torch.utils.data.DataLoader(
                    val_ds,
                    batch_size=train_batch_size,
                    sampler=ClusterSampler(val_ds.clusters, shuffle=True),
                    collate_fn=pad_collate
                )

                logger.info("Setting up model")
                model = model_class(n_features=embedding_dim, **model_config)
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

                # TODO add option to toggle off
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

                logger.info("Starting training")
                fold_progress = pbar.add_task(f"Fold {fold}", total=max_epochs)

                for epoch in range(max_epochs):
                    for split_name in split_names:
                        for metric_name in metric_names:
                            metrics[split_name][metric_name].reset()

                    train_progress = pbar.add_task(
                        f"Epoch {epoch+1} training  ", total=len(train_dl)
                    )
                    # TODO track loss per batch
                    for embs, trizod, mask in train_dl:
                        with torch.autocast(device_type=device, dtype=default_dtype):
                            model.zero_grad()
                            model.train()
                            pred = model(embs).masked_select(mask)
                            trizod = trizod.masked_select(mask)
                            loss = criterion(pred, trizod)
                    train_progress = pbar.add_task(
                        f"Epoch {epoch+1} training  ", total=len(train_dl)
                    )
                    # TODO track loss per batch
                    for embs, trizod, mask in train_dl:
                        with torch.autocast(device_type=device, dtype=default_dtype):
                            model.zero_grad()
                            model.train()
                            pred = model(embs).masked_select(mask)
                            trizod = trizod.masked_select(mask)
                            loss = criterion(pred, trizod)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                        metrics["train"]["loss"].update(
                            loss.detach(), embs.shape[0]
                        )
                        pbar.advance(train_progress)
                        metrics["train"]["loss"].update(
                            loss.detach(), embs.shape[0]
                        )
                        pbar.advance(train_progress)

                    pbar.remove_task(train_progress)

                    val_progress = pbar.add_task(
                        f"Epoch {epoch+1} validation", total=len(val_dl)
                    )

                    model.eval()
                    with torch.no_grad():
                        for embs, trizod, mask in val_dl:
                            with torch.autocast(
                                device_type=device, dtype=default_dtype
                            ):
                                pred = model(embs).masked_select(mask)
                                trizod = trizod.masked_select(mask)
                                metrics["val"]["loss"].update(
                                    criterion(pred, trizod).detach(), embs.shape[0]
                                )

                                pbar.advance(val_progress)
                        pbar.remove_task(val_progress)

                    for split_name in split_names:
                        for metric_name in metric_names:
                            foldwise_metric_values[split_name][metric_name][fold].append(metrics[split_name][metric_name].compute().item())

                    scheduler.step(foldwise_metric_values["val"]["loss"][fold][-1])
                    pbar.advance(fold_progress)
                pbar.remove_task(fold_progress)

                model_path = model_dir / f"fold_{fold}.pt"
                logging.info(f"Saving model {fold} to {model_path}")
                torch.save(model.state_dict(), model_path)

                with open(model_dir / "config.yml", "w+") as f:
                    OmegaConf.save(config=config, f=f)

                del train_ds
                del train_dl
                del val_ds
                del val_dl
                gc.collect()
                torch.cuda.empty_cache()
                pbar.advance(overall_progress)

        logger.info("Finished training, tidying up...")
        writer = SummaryWriter(log_dir="runs/" + run_name)
        for epoch in range(max_epochs):
            train_loss = np.array([foldwise_metric_values["train"]["loss"][fold][epoch] for fold in range(n_splits)]).mean()
            val_loss = np.array([foldwise_metric_values["val"]["loss"][fold][epoch] for fold in range(n_splits)]).mean()
            writer.add_scalar("loss/train", train_loss, epoch+1)
            writer.add_scalar("loss/val", val_loss, epoch+1)
    
    else:
        criterion = torch.nn.MSELoss()
        spearman = tm.regression.SpearmanCorrCoef()
        metric_names = ["loss", "spearman"]
        metrics = {metric_name: tm.aggregation.MeanMetric() for metric_name in ["loss", "spearman"]}
        writer = SummaryWriter(log_dir="runs/" + run_name)

        logger.info("Loading training data")
        train_ds = TriZodDataset(
            embedding_file,
            trizod_score_file,
            sequence_ids,
            all_cluster_assignments.filter(
                pl.col("cluster_representative_id").is_in(sequence_ids)
            ),
            device=device,
            score_type=score_type
        )
        train_dl = torch.utils.data.DataLoader(
            train_ds,
            batch_size=train_batch_size,
            sampler=ClusterSampler(train_ds.clusters, shuffle=True),
            collate_fn=pad_collate
        )

        logger.info("Setting up model")
        model = model_class(n_features=embedding_dim, **model_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # TODO add option to toggle off
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
        logger.info("Starting training")
        with progress.Progress(
            *progress.Progress.get_default_columns(), progress.TimeElapsedColumn()
        ) as pbar:
            overall_progress = pbar.add_task("Overall", total=len(train_dl))
            for epoch in range(max_epochs):
                for metric_name in metric_names:
                    metrics[metric_name].reset()

                for embs, trizod, mask in train_dl:
                    with torch.autocast(device_type=device, dtype=default_dtype):
                        model.zero_grad()
                        model.train()
                        pred = model(embs).masked_select(mask)
                        trizod = trizod.masked_select(mask)
                        loss = criterion(pred, trizod)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    metrics["loss"].update(
                        loss.detach(), embs.shape[0]
                    )
                    """
                    metrics["spearman"].update(
                        spearman(pred, trizod), embs.shape[0]
                    )
                    """

                train_loss = metrics["loss"].compute().item()
                writer.add_scalar("loss/train", train_loss, epoch+1)
                #writer.add_scalar("spearman/train", metrics["spearman"].compute().item(), epoch+1)
                scheduler.step(train_loss)
                pbar.advance(overall_progress)

        model_path = model_dir / f"final.pt"
        logging.info(f"Saving model to {model_path}")
        torch.save(model.state_dict(), model_path)

        with open(model_dir / "config.yml", "w+") as f:
            OmegaConf.save(config=config, f=f)

        del train_dl
        del train_ds
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
