#!/usr/bin/env python
"""
Copy files to this directory
"""
# %%
import os
from pathlib import Path


# %%
def read_cd_files(prefix: Path, fname: str) -> dict:
    """
    result[name] = path-to-the-file
    name: e.g., 08-1413.jpg
    pato-to-the-file: e.g., /cifs/toor/work/2011/cyclid data/./CD9/2008-06-19/08-1413.jpg
    """
    result = {}
    with open(fname) as f:
        for line in f:
            line = line.strip()
            inputf = prefix / line
            assert inputf.exists()
            result[inputf.name] = inputf
    return result


def read_paperset_file(fname) -> dict:
    """
    result[name] = fish_class
    name: e.g., 08-1413.jpg
    fish_class: e.g., lf_m
    """
    result = {}
    with open(fname) as f:
        for line in f:
            line = line.strip()
            data = line.split("/")
            assert len(data) == 3
            assert data[0] == "."
            fish_class = data[1]
            filename = data[2]
            result[filename] = fish_class
    return result



# %%
def main():
# %%
    cdfiles = read_cd_files(Path("/cifs/toor/work/2011/cyclid data"), "metadata/cd_files.txt")
# %%
    paperset = read_paperset_file("metadata/paper_set.txt")
# %%
    for name, fishclass in paperset.items():
        assert name in cdfiles
        srcfile = cdfiles[name]
        destdir = Path(f"./data/{fishclass}")
        if not destdir.exists():
            destdir.mkdir()
        os.system(f"cp -v '{str(srcfile)}' {destdir}")
# %%


if __name__ == "__main__":
    main()
