#!/usr/bin/env python
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import sklearn.model_selection
from PIL import Image  # pip install pillow

SEED = 43  # Random seed to use when splitting train/test sets.
TESTSET_SIZE = 0.2


def get_categories(df: pd.DataFrame):
    """
    Categories look like this:
    { "id": 0, "name": "gm_f" },
    { "id": 1, "name": "lf_m" }
    """
    cats = sorted(df["category"].unique().tolist())
    return [{"id": idx, "name": name} for idx, name in enumerate(cats)]


def get_wh(fname: Path) -> Tuple[int, int]:
    im = Image.open(fname)
    return im.width, im.height


def get_images(df: pd.DataFrame):
    """
    images look lie this:
    { "id": "08-0128", "file_name": "gm_f/08-0128.jpg", "width": 915, "height": 273, "location": "dummy" },
    { "id": "08-0311", "file_name": "lf_m/08-0311.jpg", "width": 1314, "height": 423, "location": "dummy" }
    """
    for _, row in df.iterrows():
        yield dict(
            id=row["id"],
            file_name=row["file_name"],
            width=row["width"],
            height=row["height"],
            location=row["location"]
        )


def get_annotations(df: pd.DataFrame, categories: List[Dict]):
    """
    annotations look like this:
    { "id": "08-0128", "image_id": "08-0128", "category_id": 0 },
    { "id": "08-0311", "image_id": "08-0311", "category_id": 1 }
    """
    lut = {item["name"]: item["id"] for item in categories}
    for id, cat_str in zip(df["id"], df["category"]):
        yield dict(id=id, image_id=id, category_id=lut[cat_str])


def make_df(files: List[Path]):

    def yield_rows(files: List[Path]):
        for fname in files:
            tmp = str(fname).split("/")
            category = tmp[1]
            id = fname.stem
            width, height = get_wh(fname)
            yield dict(
                id=id,
                file_name=f"{category}/{fname.name}",
                width=width,
                height=height,
                category=category
            )

    return pd.DataFrame(columns=["id", "file_name", "width", "height", "category"], data=yield_rows(files))


def split_train_test(df: pd.DataFrame, random_state: int, test_size: Union[float, int]) -> pd.DataFrame:
    _df_train, df_test = sklearn.model_selection.train_test_split(df, random_state=SEED, test_size=TESTSET_SIZE, stratify=df["category"])
    df["location"] = "train"  # Initially set everyting as train set.
    df.loc[df_test.index, "location"] = "test"
    return df


def make_contents(folders: List[Path], jpg_files: List[Path]):

    # df looks like this after creation:
    #           id          file_name  width  height category
    # 0    08-0128   gm_f/08-0128.jpg    915     273     gm_f
    # 1    08-0189   gm_f/08-0189.jpg   1041     342     gm_f
    # 2    08-0370   gm_f/08-0370.jpg   1059     279     gm_f
    # 3    08-0377   gm_f/08-0377.jpg    927     243     gm_f
    # 4    08-0382   gm_f/08-0382.jpg   1023     282     gm_f
    df = make_df(jpg_files)
    df = split_train_test(df, random_state=SEED, test_size=TESTSET_SIZE)  # Create location column

    categories = get_categories(df)

    contents = {
        "info": {
            "version": "0.1",
            "description": "GHOTI",
            "date_created": "2021",
            "contributor": "Deokjin Joo, Ye-seul Kwan, Jongwoo Song, Catarina Pinho, Jody Hey and Yong-Jin Won"
        },
        "categories": categories,
        "images": list(get_images(df)),
        "annotations": list(get_annotations(df, categories))
    }
    return contents


def main():
    ofname = "data/ghoti.json"

    fnames = [fname for fname in Path("./data").glob("*/*.jpg")]
    fnames.sort()
    folders = [folder for folder in Path("./data").glob("*") if folder.is_dir()]
    contents = make_contents(folders, fnames)

    with open(ofname, "w") as f:
        json.dump(contents, f, indent=2)
    print("Wrote", ofname)


if __name__ == "__main__":
    main()
