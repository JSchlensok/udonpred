import torch
from maskedtensor import masked_tensor
from sklearn.model_selection import KFold
from torchmetrics.regression import SpearmanCorrCoef
from tqdm import tqdm

from dataset.dataset import TriZodDataset
from model import TriZodModel

kfold = KFold(n_splits=5, shuffle=False)
dataset = TriZodDataset("data/train_strict.h5")

splits = list(kfold.split(dataset))

loss_func = torch.nn.MSELoss()
spearman = SpearmanCorrCoef()

models = [TriZodModel(2560, 32) for _ in range(len(splits))]

for epoch in range(10):
    for fold, (train_ids, test_ids) in enumerate(splits):
        model = models[fold]
        optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

        dataset.set_train_test(train_ids, test_ids)
        dataset.set_mode("train")

        avg_loss = 0
        avg_spearman = 0
        for embs, trizod, mask in tqdm(
            dataset, total=len(dataset), desc=f"Epoch {epoch} - Training Fold {fold}"
        ):
            model.zero_grad()
            model.train()

            pred = model(embs)

            pred = pred.masked_select(mask.bool())
            trizod = trizod.masked_select(mask.bool())
            loss = loss_func(pred, trizod)

            loss.backward()
            optimiser.step()

            avg_loss += loss.detach()
            avg_spearman += spearman(pred.detach(), trizod.detach())

        tqdm.write(f"MSE {avg_loss/len(dataset)}, Spearman {avg_spearman/len(dataset)}")

        dataset.set_mode("test")
        model.eval()
        avg_loss = 0
        avg_spearman = 0
        avg_auc = 0
        with torch.no_grad():
            for embs, trizod, mask in tqdm(
                dataset, total=len(dataset), desc=f"Epoch {epoch} - Testing Fold {fold}"
            ):
                pred = model(embs)
                loss = loss_func(pred, trizod) * mask
                pred = pred.masked_select(mask.bool())
                trizod = trizod.masked_select(mask.bool())
                avg_loss += loss_func(pred, trizod).detach()
                avg_spearman += spearman(pred, trizod).detach()

            tqdm.write(
                f"MSE {avg_loss/len(dataset)}, Spearman {avg_spearman/len(dataset)}"
            )
            tqdm.write("")
