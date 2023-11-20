import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TriZodDataset(Dataset):
    def __init__(
        self, h5_file: str, batch_size: int = 64, shuffle: bool = True
    ) -> None:
        self.data = h5py.File(h5_file, "r")
        self.length = len(self.data["cluster"])
        self.cluster_names = np.array(list(self.data["cluster"].keys()))
        self.batch_size = batch_size

        cluster_idx = torch.arange(self.length)
        if shuffle:
            indexes = torch.randperm(cluster_idx.shape[0])
            cluster_idx = cluster_idx[indexes]

        self.batches = torch.split(cluster_idx, self.batch_size)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx: int):
        cluster = self.cluster_names[self.batches[idx]]
        member = [np.random.choice(self.data["cluster"][c]) for c in cluster]

        embeddings = [
            torch.tensor(self.data["embedding"][m], dtype=torch.float32) for m in member
        ]
        trizod = [
            torch.tensor(self.data["trizod"][m], dtype=torch.float32) for m in member
        ]

        longest = max([len(e) for e in embeddings])
        padding_len = [longest - len(e) for e in embeddings]
        mask = torch.stack(
            [
                torch.concat((~t.isnan(), torch.zeros(padding_len[i])))
                for i, t in enumerate(trizod)
            ],
            0,
        )

        embeddings = torch.nn.utils.rnn.pad_sequence(embeddings).permute(1, 0, 2)
        trizod = torch.nn.utils.rnn.pad_sequence(trizod).permute(1, 0)

        return embeddings, trizod, mask
