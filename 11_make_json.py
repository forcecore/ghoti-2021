#!/usr/bin/env python
import json
from pathlib import Path
from typing import List, Tuple

from PIL import Image  # pip install pillow


def get_categories(folders: List[Path]):
    """
    Categories look like this:
    { "id": 0, "name": "gm_f" },
    { "id": 1, "name": "lf_m" }
    """
    categories = set()
    for folder in folders:
        tmp = str(folder).split("/")
        assert len(tmp) == 2
        categories.add(tmp[1])
    categories = sorted(list(categories))

    return [{"id": idx, "name": name} for idx, name in enumerate(categories)]


def get_wh(fname: Path) -> Tuple[int, int]:
    im = Image.open(fname)
    return im.width, im.height


def get_images_and_annotations(categories, fnames: List[Path]):
    """
    For each files,
    images:
    { "id": "08-0128", "file_name": "gm_f/08-0128.jpg", "width": 915, "height": 273, "location": "dummy" },
    { "id": "08-0311", "file_name": "lf_m/08-0311.jpg", "width": 1314, "height": 423, "location": "dummy" }

    annotations:
    { "id": "08-0128", "image_id": "08-0128", "category_id": 0 },
    { "id": "08-0311", "image_id": "08-0311", "category_id": 1 }
    """

    images = []
    annotations = []

    def as_image_entry(fname: Path):
        # fname looks like this: data/toc_m/08-1128.jpg
        tmp = str(fname).split("/")
        category = tmp[1]
        id = fname.stem
        width, height = get_wh(fname)
        return {"id": id, "file_name": f"{category}/{fname.name}", "width": width, "height": height, "location": "dummy", "_category": category}

    # make cat_str to cat_it lookup table
    lut = {item["name"]: item["id"] for item in categories}

    def as_annotation(x: dict):
        return { "id": x["id"], "image_id": x["id"], "category_id": lut[x["_category"]] }

    images = [as_image_entry(x) for x in fnames]
    annotations = [as_annotation(x) for x in images]
    return images, annotations


def make_contents(folders: List[Path], jpg_files: List[Path]):

    categories = get_categories(folders)
    images, annotations = get_images_and_annotations(categories, jpg_files)

    contents = {
        "info": {
            "version": "0.1",
            "description": "GHOTI",
            "date_created": "2021",
            "contributor": "Deokjin Joo, Ye-seul Kwan, Jongwoo Song, Catarina Pinho, Jody Hey and Yong-Jin Won"
        },

        "categories": categories,

        "images": images,

        "annotations": annotations
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
