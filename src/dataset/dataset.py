from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal, Union

import h5py
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from src.dataset.disprot_scores.parse import read_score_fasta
from src.dataset.trizod_scores.parse import read_score_csv


class TriZodDataset(Dataset):
    def __init__(
        self,
        embedding_file: Union[Path, str],
        score_file: Union[Path, str],
        whitelist_ids: Iterable[str],
        cluster_df: pl.DataFrame = None,
        device: str | torch.device = "cpu",
        score_type: Literal["trizod", "chezod"] = "trizod"
    ) -> None:
        score_column_name = "pscores" if score_type == "trizod" else "zscores"
        scores = (
            read_score_csv(score_file).group_by(pl.col("ID")).agg(pl.col(score_column_name))
        )
        cluster_df = cluster_df.filter(
            pl.col("cluster_representative_id").is_in(whitelist_ids)
        )
        self.cluster_representative_ids = (
            cluster_df.select("cluster_representative_id").to_series().to_list()
        )
        self.all_ids = cluster_df.select("sequence_id").to_series().to_list()
        if cluster_df is not None:
            self.clusters = {
                row[0]: row[1]
                for row in cluster_df.group_by("cluster_representative_id")
                .agg(pl.col("sequence_id"))
                .iter_rows()
            }
        else:
            self.clusters = None

        self.scores = {
            row[0]: torch.from_numpy(np.array(row[1], dtype=np.float32)).to(device)
            for row in scores.filter(pl.col("ID").is_in(self.all_ids)).iter_rows()
        }
        self.embeddings = {
            id: torch.from_numpy(np.array(emb[()], dtype=np.float32)).to(device) for id, emb in h5py.File(embedding_file).items()
            if id in self.all_ids
        }
        self.nan_masks = {id: ~score.isnan() for id, score in self.scores.items()}
        self.indices = {i: id for i, id in enumerate(self.all_ids)}

    def __len__(self):
        return len(self.cluster_representative_ids)

    def __getitem__(self, id: str):
        if isinstance(id, int):
            id = self.indices[id]
        embedding = self.embeddings[id].clone()
        scores = self.scores[id].clone()
        mask = self.nan_masks[id].clone()
        return embedding, scores, mask


class DisprotDataset(Dataset):
    def __init__(
        self,
        embedding_file: Union[Path, str],
        score_file: Path,
        device: str | torch.device,
    ) -> None:
        annotated_sequences = read_score_fasta(score_file)
        self.scores = {id: torch.from_numpy(np.array(x.annotations, dtype=np.float16)).to(device) for id, x in annotated_sequences.items()}
        self.all_ids = list(annotated_sequences.keys())
        self.embeddings = {
            id: torch.from_numpy(np.array(emb[()], dtype=np.float32)).to(device) for id, emb in h5py.File(embedding_file).items()
            if id in self.all_ids
        }
        self.nan_masks = {id: ~score.isnan() for id, score in self.scores.items()}
        self.indices = {i: id for i, id in enumerate(self.all_ids)}

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, id: str | int):
        if isinstance(id, int):
            id = self.indices[id]
        embedding = self.embeddings[id].clone()
        scores = self.scores[id].clone()
        mask = self.nan_masks[id].clone()
        return embedding, scores, mask
