"""Generate SHA256 manifests for the frozen repository source and empirical data."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

DATA_FILES = [
    Path("networks_data/lesmis/lesmis.mtx"),
    Path("networks_data/adjnoun/adjnoun.mtx"),
    Path("networks_data/jazz/jazz.mtx"),
    Path("networks_data/USAir97/USAir97.mtx"),
    Path("networks_data/ia-infect-dublin.mtx"),
    Path("networks_data/email/email.mtx"),
    Path("networks_data/polblogs/polblogs.mtx"),
    Path("networks_data/soc-hamsterster.edges"),
    Path("networks_data/power/power.mtx"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(paths, target):
    lines = []
    for rel in paths:
        path = ROOT / rel
        if path.exists():
            lines.append(f"{sha256(path)}  {rel.as_posix()}")
    (DOCS / target).write_text("\n".join(lines) + "\n", encoding="utf-8")


def source_files():
    excluded_dirs = {".git", "__pycache__", "results"}
    excluded_names = {
        "dataset_file_checksums_sha256.txt",
        "repository_source_checksums_sha256.txt",
    }
    items = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if any(part in excluded_dirs for part in rel.parts):
            continue
        if p.name in excluded_names:
            continue
        items.append(rel)
    return sorted(items, key=lambda x: x.as_posix())


def main():
    DOCS.mkdir(exist_ok=True)
    write_manifest(DATA_FILES, "dataset_file_checksums_sha256.txt")
    write_manifest(source_files(), "repository_source_checksums_sha256.txt")
    print("Wrote checksum manifests under docs/.")


if __name__ == "__main__":
    main()
