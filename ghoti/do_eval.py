"""
Evaluation script
"""
import json
import sys
from pathlib import Path

import sklearn.metrics
import torch
import torchvision as tv
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .model import MyResnet
from .util import dump_as_json, load_pickle


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
    result = dict(
        accuracy=accuracy,
        f1=f1
    )
    return result


def check_input(cnf: DictConfig, cmd: DictConfig):
    assert "test_set_file" in cmd
    assert "weight_file" in cmd
    assert "class_to_index_file" in cmd
    assert "eval_result" in cmd

    assert "batch_size" in cnf
    assert "num_workers" in cnf


def main(argv: "list[str]"):
    """
    Do evaluation
    """
    cnf = OmegaConf.load(argv[0])
    cmd = OmegaConf.from_dotlist(argv[1:])
    check_input(cnf, cmd)

    with Path(cmd.class_to_index_file).open() as f:
        class_to_idx: dict = json.load(f)
    test_set: tv.datasets.ImageFolder = load_pickle(cmd.test_set_file)

    net = MyResnet(cnf=None, num_classes=len(class_to_idx))
    sd = torch.load(cmd.weight_file)
    net.load_state_dict(sd)

    eval_result = do_eval(cnf, net, test_set)
    print(eval_result)
    dump_as_json(eval_result, cmd.eval_result)
    print("Wrote", cmd.eval_result)


if __name__ == "__main__":
    main(sys.argv[1:])
