import warnings
from pathlib import Path
from typing import Annotated, Literal

import hydra
import mlflow
import numpy as np
import polars as pl
import torch
import torchmetrics as tm
import typer
from Bio import SeqIO
from flatdict import FlatDict
from maskedtensor import masked_tensor
from omegaconf import DictConfig
from rich import progress
from sklearn.model_selection import KFold
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

from src.dataset.clustering.parse import read_cluster_assignments
from src.dataset.dataset import TriZodDataset
from src.dataset.utils import ClusterSampler, pad_collate
from src.models import FNN, SETHClone
from src.utils.logging import setup_logger

embedding_dimensions = {"prott5": 1024, "esm2_3b": 2560}

model_classes = {"fnn": FNN, "cnn": SETHClone}


# TODO parametrize config name
@hydra.main(version_base=None, config_path="../../parameters", config_name="linreg")
def main(config: DictConfig):
    logger = setup_logger()
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

    logger.info("Parsing parameters")
    subset = config.data.subset
    embedding_type = config.data.embedding_type
    embedding_dim = embedding_dimensions[embedding_type]

    train_batch_size = config.training.batch_size
    learning_rate = config.training.learning_rate
    n_splits = config.training.n_splits
    max_epochs = config.training.max_epochs

    val_batch_size = config.validation.batch_size

    model_class = model_classes[config.model.type]
    model_config = config.model.params

    # TODO parametrize
    project_root = Path.cwd()
    mlflow_dir = project_root / "runs"
    model_dir = project_root / "models"

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
        f"Using device {device} and automatic mixed precision wit default dtype {default_dtype}"
    )

    torch.set_default_device(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # TODO add option for training only one model
    logger.info("Initializing dataset")
    all_cluster_assignments = read_cluster_assignments(
        f"data/clusters/{subset}_rest_clu.tsv"
    )
    embedding_file = f"data/embeddings/unfiltered_all_{embedding_type}_res.h5"
    trizod_score_file = f"data/{subset}.csv"
    sequence_ids = np.array(
        all_cluster_assignments.select("sequence_id").to_series(), dtype=str
    )

    kfold = KFold(n_splits=n_splits, shuffle=False)
    splits = list(kfold.split(sequence_ids))
    logger.info("Loading training data")
    train_datasets = [
        TriZodDataset(
            embedding_file,
            trizod_score_file,
            sequence_ids[train_indices],
            all_cluster_assignments.filter(
                pl.col("cluster_representative_id").is_in(sequence_ids[train_indices])
            ),
        )
        for train_indices, _ in tqdm(splits, desc="Loading training data", leave=False)
    ]
    train_dataloaders = [
        torch.utils.data.DataLoader(
            ds,
            batch_size=train_batch_size,
            sampler=ClusterSampler(ds.clusters, shuffle=True),
            collate_fn=pad_collate,
        )
        for ds in train_datasets
    ]
    logger.info("Loading validation data")
    val_datasets = [
        TriZodDataset(
            embedding_file,
            trizod_score_file,
            sequence_ids[val_indices],
            all_cluster_assignments.filter(
                pl.col("cluster_representative_id").is_in(sequence_ids[val_indices])
            ),
        )
        for _, val_indices in tqdm(splits, desc="Loading validation data", leave=False)
    ]
    val_dataloaders = [
        torch.utils.data.DataLoader(
            ds,
            batch_size=val_batch_size,
            sampler=ClusterSampler(ds.clusters, active=False),
            collate_fn=pad_collate,
        )
        for ds in val_datasets
    ]

    logger.info("Setting up models")
    models = [
        model_class(n_features=embedding_dim, **model_config) for _ in range(n_splits)
    ]
    optimizers = [
        torch.optim.AdamW(model.parameters(), lr=learning_rate) for model in models
    ]

    # TODO add option for disabling Spearman computation on validation set
    # Set up metrics
    spearman = tm.regression.SpearmanCorrCoef()

    criterion = torch.nn.MSELoss()

    # TODO parametrize
    aggregation_levels = ["epoch", "fold"]
    split_names = ["train", "val"]
    metric_names = ["loss", "spearman"]
    metrics = {
        aggregation_level: {
            split: {metric: tm.aggregation.MeanMetric() for metric in metric_names}
            for split in split_names
        }
        for aggregation_level in aggregation_levels
    }

    mlflow.set_tracking_uri("file://" + str(mlflow_dir))
    mlflow
    with mlflow.start_run() as run:
        with progress.Progress(
            *progress.Progress.get_default_columns(), progress.TimeElapsedColumn()
        ) as pbar:
            # TODO log dataset as artifact
            overall_progress = pbar.add_task("Overall", total=max_epochs)
            for epoch in range(max_epochs):
                for split_name, metric_name in zip(split_names, metric_names):
                    metrics["epoch"][split_name][metric_name].reset()

                epoch_progress = pbar.add_task(f"Epoch {epoch}", total=n_splits)

                for fold in range(n_splits):
                    model = models[fold]
                    optimizer = optimizers[fold]
                    train_dl = train_dataloaders[fold]
                    val_dl = val_dataloaders[fold]

                    train_progress = pbar.add_task(
                        f"Fold {fold} training  ", total=len(train_dl)
                    )
                    for split_name, metric_name in zip(split_names, metric_names):
                        metrics["fold"][split_name][metric_name].reset()

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

                        metrics["fold"]["train"]["loss"].update(
                            loss.detach(), embs.shape[0]
                        )
                        pbar.advance(train_progress)

                    pbar.remove_task(train_progress)

                    val_progress = pbar.add_task(
                        f"Fold {fold} validation", total=len(val_dl)
                    )

                    model.eval()
                    with torch.no_grad():
                        for embs, trizod, mask in val_dl:
                            with torch.autocast(
                                device_type=device, dtype=default_dtype
                            ):
                                pred = model(embs).masked_select(mask)
                                trizod = trizod.masked_select(mask)
                                batch_size = embs.shape[0]
                                metrics["fold"]["val"]["loss"].update(
                                    criterion(pred, trizod).detach(), batch_size
                                )
                                metrics["fold"]["val"]["spearman"].update(
                                    spearman(pred, trizod).detach(), batch_size
                                )

                            pbar.advance(val_progress)
                        pbar.remove_task(val_progress)

                    for split_name in split_names:
                        for metric_name in metric_names:
                            metrics["epoch"][split_name][metric_name].update(
                                metrics["fold"][split_name][metric_name].compute()
                            )
                    pbar.advance(epoch_progress)

                mlflow.log_metric(
                    "train_loss",
                    metrics["epoch"]["train"]["loss"].compute().item(),
                    step=epoch,
                )
                mlflow.log_metric(
                    "val_loss",
                    metrics["epoch"]["val"]["loss"].compute().item(),
                    step=epoch,
                )
                mlflow.log_metric(
                    "val_spearman",
                    metrics["epoch"]["val"]["spearman"].compute().item(),
                    step=epoch,
                )
                pbar.remove_task(epoch_progress)
                pbar.advance(overall_progress)

        logger.info("Finished training, tidying up...")
        mlflow.log_params(FlatDict(config, delimiter='.'))
        # TODO save models


if __name__ == "__main__":
    main()
