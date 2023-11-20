from sklearn.model_selection import KFold

from dataset.dataset import TriZodDataset

kfold = KFold(n_splits=5, shuffle=False)
dataset = TriZodDataset("data/train_strict.h5")

splits = list(kfold.split(dataset))

for fold, (train_ids, test_ids) in enumerate(splits):
    print(train_ids, test_ids)
    dataset.set_train_test(train_ids, test_ids)
    dataset.set_mode("train")

    print(f"Training on fold {fold}...")
    for embs, trizod, mask in iter(dataset):
        print(embs.shape, trizod.shape, mask.shape)

    print(f"Testing on fold {fold}...")
    dataset.set_mode("test")
    for embs, trizod, mask in dataset:
        print(embs.shape, trizod.shape, mask.shape)
