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

        self.train = np.arange(len(self.batches))
        self.test = np.array([])
        self.train_iter = iter(self.train)
        self.test_iter = iter(self.test)
        self.is_training = True

    def __len__(self):
        return len(self.train) if self.is_training else len(self.test)

    def __getitem__(self, idx: int):
        cluster = self.cluster_names[
            self.batches[
                next(self.train_iter) if self.is_training else next(self.test_iter)
            ]
        ]
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

    def set_train_test(self, train: np.array, test: np.array):
        self.train = train
        self.test = test
        self.train_iter = iter(train)
        self.test_iter = iter(test)

    def set_mode(self, mode: str):
        self.is_training = True if mode == "train" else False
