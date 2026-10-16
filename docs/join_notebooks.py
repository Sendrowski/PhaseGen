"""
Join the executed notebooks of the parts of a User Guide page in one language into the executed notebook of the page,
by concatenating their cells in order. The parts are executed independently, so that a change to one part re-executes
only that part.
"""
import json
import sys
from pathlib import Path


def join(parts: list[Path], out: Path) -> None:
    """
    Concatenate the cells of notebooks.

    :param parts: The notebooks, in order.
    :param out: The joined notebook, which takes the metadata of the first.
    """
    notebooks = [json.loads(part.read_text()) for part in parts]
    joined = dict(notebooks[0], cells=[cell for notebook in notebooks for cell in notebook["cells"]])
    out.write_text(json.dumps(joined, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    try:
        join([Path(p) for p in snakemake.input], Path(snakemake.output[0]))
    except NameError:
        join([Path(p) for p in sys.argv[2:]], Path(sys.argv[1]))
