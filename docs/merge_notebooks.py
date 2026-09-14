"""
Merge a paired Python and R documentation notebook into one page whose code is shown in synchronised language tabs.

Both notebooks hold the same sequence of shared markdown cells. The page takes its prose from the Python notebook, and
the code cells between two shared markdown cells become one tab per language, with the executed outputs pasted by
myst-nb glue from hidden carrier cells in the merged notebook. A markdown cell tagged ``r-only`` in the R notebook (or
``python-only`` in the Python notebook) opens a segment whose prose and code are rendered inside that language's tab,
after the shared segment it follows. The language-only segments of both notebooks at one position share a tab set, so
their markdown must not contain section headings.

The notebooks are split from the page source ``docs/source/{name}.md`` by ``docs/split_page.py`` and executed
before merging. The Snakemake rule ``merge_page`` writes ``docs/reference/{name}.ipynb`` from
``results/docs/Python/{name}.executed.ipynb`` and ``results/docs/R/{name}.executed.ipynb``. Run directly,
``python docs/merge_notebooks.py <name> ...`` does the same for each page name.
"""
import base64
import copy
import difflib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

GLUE_PREFIX = "application/papermill.record/"

LANGUAGES = {
    "Python": dict(tab="{fab}`python` Python", sync="python", lexer="python", only="python-only"),
    "R": dict(tab="{fab}`r-project` R", sync="r", lexer="r", only="r-only"),
}

# class of the language tab sets, styled in docs/_static/custom.css (docutils strips classes beginning with "language-")
TAB_SET_CLASS = "code-tabs"


def tags(cell: dict) -> list:
    return cell.get("metadata", {}).get("tags", [])


def text(cell: dict) -> str:
    return "".join(cell["source"])


def split_segments(nb: dict, only_tag: str) -> tuple[list, list]:
    """
    Split a notebook into shared segments and language-only segments.

    :return: ``shared``, a list of ``{"md": cell, "code": [cells]}``, and ``only``, a list of
        ``(shared_index, segment)`` for segments opened by a markdown cell tagged ``only_tag``.
    """
    shared, only = [], []
    current = None

    for cell in nb["cells"]:
        if "remove-cell" in tags(cell):
            continue

        if cell["cell_type"] == "markdown":
            current = {"md": cell, "code": []}

            if only_tag in tags(cell):
                only.append((len(shared) - 1, current))
            else:
                shared.append(current)

        elif cell["cell_type"] == "code":
            if current is None:
                current = {"md": None, "code": []}
                shared.append(current)

            current["code"].append(cell)

    return shared, only


# resolution both languages render figures at, set in the setup cells of docs/source/*.md
FIGURE_DPI = 300

# figures displayed wider than this are multi-panel grids, shown at their full width
SINGLE_FIGURE_MAX_WIDTH = 600

# display scale of single-panel figures
SINGLE_FIGURE_SCALE = 0.8


def display_metadata(data: dict, metadata: dict) -> dict:
    """
    Display metadata of an output. A PNG image is shown at its nominal size of 100 CSS pixels per inch rendered at
    ``FIGURE_DPI``, shrunk by ``SINGLE_FIGURE_SCALE`` for a single-panel figure. Only the width is given, as myst-nb
    writes a given height as an inline style that holds while the column scales the width down.
    """
    metadata = copy.deepcopy(metadata)

    if "image/png" in data:
        pixels = struct.unpack(">I", base64.b64decode(data["image/png"])[16:20])[0]
        width = round(pixels * 100 / FIGURE_DPI)

        if width <= SINGLE_FIGURE_MAX_WIDTH:
            width = round(width * SINGLE_FIGURE_SCALE)

        metadata["image/png"] = dict(width=width)

    return metadata


def carrier_outputs(cell: dict, key_prefix: str) -> tuple[list, list]:
    """
    Convert a code cell's outputs into hidden glue outputs.

    :return: The glue outputs and their keys, in display order.
    """
    outputs, keys = [], []

    if "remove-output" in tags(cell):
        return outputs, keys

    for output in cell.get("outputs", []):
        if output["output_type"] == "stream":
            if outputs and outputs[-1]["stream"] == output["name"]:
                outputs[-1]["data"][GLUE_PREFIX + "text/plain"] += "".join(output["text"])
                continue
            data = {"text/plain": "".join(output["text"])}
            stream = output["name"]
        elif output["output_type"] in ("display_data", "execute_result"):
            data = copy.deepcopy(output["data"])
            # IRkernel pairs every value with HTML, Markdown and LaTeX renderings; the plain text matches the Python tab
            if "text/plain" in data and not any(k.startswith("image/") for k in data):
                data = {"text/plain": data["text/plain"]}
            stream = None
        else:
            continue

        key = f"{key_prefix}-{len(keys)}"
        keys.append(key)
        outputs.append(dict(
            output_type="display_data",
            data={GLUE_PREFIX + k: v for k, v in data.items()},
            metadata=dict(
                display_metadata(data, output.get("metadata", {})),
                scrapbook=dict(name=key, mime_prefix=GLUE_PREFIX),
            ),
            stream=stream,
        ))

    for output in outputs:
        del output["stream"]

    return outputs, keys


def fence(n: int, directive: str, argument: str, options: str, body: str) -> str:
    # tilde fences, as the info string of a backtick fence cannot contain the backticks of an icon role
    tildes = "~" * n
    return f"{tildes}{{{directive}}} {argument}\n{options}\n\n{body}\n{tildes}\n"


def tab_block(code_by_language: dict, carriers: list, segment_id: str) -> str:
    """Render one segment's code as a tab set, appending the hidden glue carrier cells to ``carriers``."""
    items = []

    for language, cells in code_by_language.items():
        spec = LANGUAGES[language]
        parts = []

        for i, cell in enumerate(cells):
            if cell["cell_type"] == "markdown":
                parts.append(text(cell).rstrip() + "\n")
                continue

            if "remove-input" not in tags(cell) and text(cell).strip():
                parts.append(f"```{spec['lexer']}\n{text(cell).rstrip()}\n```\n")

            outputs, keys = carrier_outputs(cell, f"{spec['sync']}-{segment_id}-{i}")
            parts += [f"```{{glue}} {key}\n```\n" for key in keys]

            if outputs:
                carriers.append(dict(cell_type="code", execution_count=None, source=[],
                                     metadata=dict(tags=["remove-cell"]), outputs=outputs))

        if parts:
            items.append(fence(5, "tab-item", spec["tab"], f":sync: {spec['sync']}", "\n".join(parts)))

    if not items:
        return ""

    return fence(6, "tab-set", "", f":sync-group: language\n:class: {TAB_SET_CLASS}", "\n".join(items))


def markdown_cell(source: str) -> dict:
    lines = source.split("\n")
    return dict(cell_type="markdown", metadata={}, source=[l + "\n" for l in lines[:-1]] + [lines[-1]])


def merge(python_path: Path, r_path: Path, out: Path):
    """Merge the Python notebook at ``python_path`` and the R notebook at ``r_path`` into ``out``."""
    python = json.load(open(python_path))
    r = json.load(open(r_path))
    page = out.stem

    py_shared, py_only = split_segments(python, LANGUAGES["Python"]["only"])
    r_shared, r_only = split_segments(r, LANGUAGES["R"]["only"])

    if len(py_shared) != len(r_shared):
        sys.exit(f"{page}: {len(py_shared)} shared Python segments but {len(r_shared)} shared R segments")

    for k, (a, b) in enumerate(zip(py_shared, r_shared)):
        ta, tb = (text(s["md"]) if s["md"] else "" for s in (a, b))
        ratio = difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()
        if ratio < 1:
            print(f"{page}: shared markdown {k} differs between Python and R (similarity {ratio:.2f}); "
                  f"using the Python prose")

    cells, carriers = [], []
    extra = {k: [] for k in range(-1, len(py_shared))}
    for language, only in (("Python", py_only), ("R", r_only)):
        for k, segment in only:
            extra[k].append((language, segment))

    def emit(md, code_by_language, segment_id):
        if md is not None:
            cells.append(markdown_cell(text(md)))
        block = tab_block(code_by_language, carriers, segment_id)
        if block:
            cells.append(markdown_cell(block))

    def emit_only(k):
        tabs = {}
        for language, segment in extra[k]:
            tabs.setdefault(language, []).extend(([segment["md"]] if segment["md"] else []) + segment["code"])
        emit(None, tabs, f"s{k}-only")

    emit_only(-1)
    for k, (a, b) in enumerate(zip(py_shared, r_shared)):
        emit(a["md"], {"Python": a["code"], "R": b["code"]}, f"s{k}")
        emit_only(k)

    merged = dict(nbformat=4, nbformat_minor=5, metadata=python["metadata"], cells=cells + carriers)
    out.write_text(json.dumps(merged, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    try:
        merge(Path(snakemake.input.python), Path(snakemake.input.r), Path(snakemake.output[0]))
    except NameError:
        for name in sys.argv[1:]:
            merge(ROOT / "results" / "docs" / "Python" / f"{name}.executed.ipynb",
                  ROOT / "results" / "docs" / "R" / f"{name}.executed.ipynb",
                  ROOT / "docs" / "reference" / f"{name}.ipynb")
