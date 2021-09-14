"""
Training script for the project
"""
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plcb
import sklearn.metrics
import torch
import torchvision as tv
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .model import MyResnet


def get_datasets(cnf):
    """
    Returns train and test sets.
    If I were to do this properly, I'd need validation set too.
    But for the sake of brevity....
    """
    transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Resize((cnf.nn_input_size, cnf.nn_input_size))
    ])

    dataset = tv.datasets.ImageFolder(
        root=cnf.dataset_root,
        transform=transforms
    )
    # (1) Use dataset.class_to_idx to see the mappings.
    #     For example, you'll get
    #     {'gm_f': 0, 'lf_m': 1, 'mv_f': 2, 'pe_m': 3, 'pf_f': 4, 'pg_f': 5, 'tg_f': 6, 'tg_m': 7, 'tm_f': 8, 'tm_m': 9, 'toc_f': 10, 'toc_m': 11}
    # (2) for x, y in train_dataset:
    #     print(x.shape)
    #     print(y)
    # print(len(dataset))
    # Without Resize transform, you get something like
    # x: torch.Size([3, 440, 1320])
    # y: (int)
    ds_size = len(dataset)
    test_size = int(cnf.test_ratio * ds_size)
    train_set, test_set = torch.utils.data.random_split(
                                dataset,
                                [ds_size - test_size, test_size],
                                generator=torch.Generator().manual_seed(cnf.split_seed))
    return train_set, test_set


def do_training(cnf, net, train_set):
    # Pretrain the FC layer before doing the full training
    net.freeze()
    for parameters in net.resnet.fc.parameters():  # resnet.fc doesn't have unfreeze() so...
        parameters.requires_grad = True

    # Train only the last FC layer
    train_loader = DataLoader(train_set, batch_size=cnf.batch_size, shuffle=True, num_workers=cnf.num_workers)
    trainer = pl.Trainer(gpus=cnf.ngpus, max_epochs=cnf.max_epochs,
                    default_root_dir="./checkpoints",
                    callbacks=[
                        plcb.EarlyStopping(monitor="train_loss", patience=cnf.patience)  # stop when train_loss plateaus
                    ])
    trainer.fit(net, train_loader)

    # Unfreeze everything
    # Nay, I think the above training is enough!
    # net.unfreeze()
    # trainer = pl.Trainer(gpus=1, max_epochs=100, default_root_dir="./checkpoints", callbacks=[checkpoint_callback])
    # trainer.fit(net, train_loader)


def as_indices(y_preds: "list[torch.Tensor]") -> "list[int]":
    """
    y_preds is a list of matrices, where mtx each row is a prediction for one sample
    and the column represents the "probability" for each class.
    We pick one most probable class and return them as a list for all predictions.
    """
    result = []
    for y_pred in y_preds:
        idxs = torch.argmax(y_pred, axis=1)
        for clsid in idxs:
            result.append(clsid)
    return result


def do_eval(cnf, net, test_set):
    test_loader = DataLoader(test_set, batch_size=cnf.batch_size, shuffle=False, num_workers=cnf.num_workers)
    y_preds = []
    y_true = []
    for batch in test_loader:
        x, y = batch
        y_pred = net.forward(x)

        # Convert predictions into class indices
        y_preds.append(y_pred)
        for clsid in y:
            y_true.append(clsid)

    clsid_pred = as_indices(y_preds)

    accuracy = sklearn.metrics.accuracy_score(y_true, clsid_pred)
    f1 = sklearn.metrics.f1_score(y_true, clsid_pred, average="macro")
    print(f"Accuracy: {accuracy}")
    print(f"f1: {f1}")


def main():
    """
    Train the model
    """
    cnf = OmegaConf.load("config.yaml")
    net = MyResnet(num_classes=12)
    train_set, test_set = get_datasets(cnf)

    do_training(cnf, net, train_set)

    # Write snapshot
    ofname = "ghoti.pt"
    torch.save(net.state_dict(), ofname)
    print("Training complete! Wrote", ofname)

    # Do evaluation
    do_eval(cnf, net, test_set)


if __name__ == "__main__":
    main()
