"""
Utility functions
"""
import json
import pickle
from pathlib import Path


def maybe_make_output_dir(ofname: "Path|str"):
    if isinstance(ofname, str):
        ofname = Path(ofname)
    odir = ofname.parent
    if not odir.exists():
        odir.mkdir(parents=True, exist_ok=True)


def pickle_object(obj, ofname: "Path|str"):
    """
    Pickle given object as ofname, create directory of needed.
    """
    ofname = Path(ofname)
    maybe_make_output_dir(ofname)
    with ofname.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(input_file: str):
    fname = Path(input_file)
    assert fname.exists()
    with fname.open("rb") as f:
        return pickle.load(f)


def dump_as_json(obj, ofname: "Path|str", indent=2):
    """
    Pickle given object as ofname, create directory of needed.
    """
    ofname = Path(ofname)
    maybe_make_output_dir(ofname)
    with ofname.open("w") as f:
        json.dump(obj, f, indent=indent)
