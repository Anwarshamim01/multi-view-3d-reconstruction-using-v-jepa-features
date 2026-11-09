"""Check the two final notebooks without downloading data or starting training."""
import ast
import base64
import hashlib
import io
from pathlib import Path
import zipfile

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "kaggle_v44": "train_tum_rgbd_2gpu.ipynb",
    "kaggle_v46": "train_co3d_2gpu.ipynb",
}


def main():
    for package, filename in NOTEBOOKS.items():
        path = ROOT / package / filename
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        bundle = {}
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            assert cell.execution_count is None and not cell.outputs, (path, index)
            tree = ast.parse(cell.source, filename=f"{path.name}:cell{index}")
            compile(tree, f"{path.name}:cell{index}", "exec")
            for node in tree.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target = node.targets[0]
                    if isinstance(target, ast.Name) and target.id in {"BUNDLE_BASE64", "BUNDLE_SHA256"}:
                        bundle[target.id] = ast.literal_eval(node.value)
        payload = base64.b64decode(bundle["BUNDLE_BASE64"])
        assert hashlib.sha256(payload).hexdigest() == bundle["BUNDLE_SHA256"], path
        expected = {f"{package}/{p.name}" for p in (ROOT / package).glob("*.py")}
        expected.add(f"{package}/model_defaults.json")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            assert set(archive.namelist()) == expected, path
            for name in archive.namelist():
                assert archive.read(name) == (ROOT / name).read_bytes(), name
        print("PASS", path.relative_to(ROOT), "cells compile; embedded source matches Python modules")


if __name__ == "__main__":
    main()
