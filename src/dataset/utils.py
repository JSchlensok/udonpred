import numpy as np
import torch


class ClusterSampler(torch.utils.data.Sampler):
    def __init__(
        self, clusters: dict[str, list[str]], shuffle: bool = False, active: bool = True
    ):
        self.clusters = clusters
        self.shuffle = shuffle
        self.active = active

    def __iter__(self):
        cluster_representatives = list(self.clusters.keys())
        if self.shuffle:
            # using Numpy to have same seed as all other random operations that use it
            np.random.shuffle(cluster_representatives)
        for rep_id in cluster_representatives:
            if self.active:
                yield np.random.choice(self.clusters[rep_id])
            else:
                yield rep_id

    def __len__(self):
        return len(self.clusters)


def pad_collate(batch):
    embeddings, scores, nan_masks = zip(*batch)
    embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
    scores = torch.nn.utils.rnn.pad_sequence(scores, batch_first=True)
    nan_masks = torch.nn.utils.rnn.pad_sequence(
        nan_masks, padding_value=False, batch_first=True
    )

    return embeddings, scores, nan_masks
