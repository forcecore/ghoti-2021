"""
Training script for the project
"""
import json
import sys
from pathlib import Path

import pytorch_lightning as pl
import pytorch_lightning.callbacks as plcb
import torch
import torchvision as tv
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .model import MyResnet
from .util import load_pickle


def do_training(cnf, net: MyResnet, train_set: tv.datasets.ImageFolder):
    # Pretrain the FC layer before doing the full training
    net.freeze()
    for parameters in net.resnet.fc.parameters():  # resnet.fc doesn't have unfreeze() so...
        parameters.requires_grad = True

    # Train only the last FC layer
    train_loader = DataLoader(train_set, batch_size=cnf.data_loader.batch_size, shuffle=True, num_workers=cnf.data_loader.num_workers)
    trainer = pl.Trainer(gpus=cnf.train.ngpus, max_epochs=cnf.train.max_epochs,
                    default_root_dir="./checkpoints",
                    callbacks=[
                        plcb.EarlyStopping(monitor="train_loss", patience=cnf.train.patience)  # stop when train_loss plateaus
                    ])
    trainer.fit(net, train_loader)

    # Unfreeze everything
    # Nay, I think the above training is enough!
    # net.unfreeze()
    # trainer = pl.Trainer(gpus=1, max_epochs=100, default_root_dir="./checkpoints", callbacks=[checkpoint_callback])
    # trainer.fit(net, train_loader)


def check_input(cmd: DictConfig):
    assert "train_set_file" in cmd
    assert "weight_file" in cmd


def main(argv: "list[str]"):
    """
    Train the model
    """
    cnf = OmegaConf.load("params.yaml")
    cmd = OmegaConf.from_dotlist(argv)
    check_input(cmd)

    train_set: tv.datasets.ImageFolder = load_pickle(cmd.train_set_file)
    with Path(cmd.class_to_index_file).open() as f:
        class_to_idx: dict = json.load(f)
    net = MyResnet(num_classes=len(class_to_idx), learning_rate=cnf.train.learning_rate)
    do_training(cnf, net, train_set)

    # Write snapshot
    torch.save(net.state_dict(), cmd.weight_file)
    print("Training complete! Wrote", cmd.weight_file)


if __name__ == "__main__":
    main(sys.argv[1:])
