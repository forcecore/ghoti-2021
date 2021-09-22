import pickle
import sys
from pathlib import Path

import torch
import torchvision as tv
from omegaconf.omegaconf import DictConfig, OmegaConf
from .util import dump_as_json, pickle_object


def get_datasets(dataset_root: "str|Path", nn_input_size: int, test_ratio: float, split_seed: int):
    """
    Returns train and test sets.
    If I were to do this properly, I'd need validation set too.
    But for the sake of brevity...

    dataset_root: root directory of the dataset
    nn_input_size: size of images the neural network expects
    test_ratio: A number in range (0, 1). We will use this much samples as the test set.
    split_seed: random seed to be used in the dataset split process.
    """
    transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Resize((nn_input_size, nn_input_size))
    ])

    dataset = tv.datasets.ImageFolder(
        root=dataset_root,
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
    test_size = int(test_ratio * ds_size)
    train_set, test_set = torch.utils.data.random_split(
                                dataset,
                                [ds_size - test_size, test_size],
                                generator=torch.Generator().manual_seed(split_seed))
    return train_set, test_set, dataset.class_to_idx


def check_input(cnf: DictConfig, cmd: DictConfig):
    assert "dataset_root" in cnf
    assert "nn_input_size" in cnf
    assert "test_ratio" in cnf
    assert "split_seed" in cnf

    assert "train_set_file" in cmd
    assert "test_set_file" in cmd


def main(argv: "list[str]"):
    cnf = OmegaConf.load(argv[0])
    cmd = OmegaConf.from_dotlist(argv[1:])
    check_input(cnf, cmd)

    train_set, test_set, class_to_index = get_datasets(
        dataset_root=cnf.dataset_root,
        nn_input_size=cnf.nn_input_size,
        test_ratio=cnf.test_ratio,
        split_seed=cnf.split_seed
    )

    pickle_object(train_set, cmd.train_set_file)
    print("Wrote", cmd.train_set_file)
    pickle_object(test_set, cmd.test_set_file)
    print("Wrote", cmd.test_set_file)
    dump_as_json(class_to_index, cmd.class_to_index_file)
    print("Wrote", cmd.class_to_index_file)


if __name__ == "__main__":
    main(sys.argv[1:])
