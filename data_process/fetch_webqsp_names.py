"""Step 0: fetch the two MID->name sources for WebQSP into data/raw/webqsp_names/.

Sources (pinned by sha256):
  * GNN-RAG ``entities_names.json`` (560,012 m.* MIDs -> English name), Google Drive
    folder linked from github.com/cmavro/GNN-RAG (gnn/README.md, llm/README.md).
  * HF ``rmanluo/RoG-webqsp`` parquet shards: the dense NSM webqsp subgraphs with
    node names filled in (same triples, same order), used to recover extra names.

Usage:
  python data_process/fetch_webqsp_names.py                 # download what is missing
  python data_process/fetch_webqsp_names.py --copy-from DIR # copy local copies instead
  python data_process/fetch_webqsp_names.py --verify-only
Every file is checked against its pinned sha256; a mismatch is an error.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import NAMES_DIR, ensure_dir, log, sha256_file  # noqa: E402

HF = "https://huggingface.co/datasets/rmanluo/RoG-webqsp/resolve/main/data/"
FILES = {
    "entities_names.json": (
        "https://drive.usercontent.google.com/download?id=1H7BqbVsQXr0bGuBsLUphb_Cb0Jf0rSoN&export=download",
        "93ce094acccb06b1906db8a4534175481efe55bb9fa8ed711fedd1f8b37d2f98", 23377816),
    "rog/train-00000-of-00002-d810a36ed97bc2cc.parquet": (
        HF + "train-00000-of-00002-d810a36ed97bc2cc.parquet",
        "4d2d4520cffb87370fa7de408dc6730a1e8d90d0233e166117a377ca71b5f2e8", 154472902),
    "rog/train-00001-of-00002-e53244e71082a392.parquet": (
        HF + "train-00001-of-00002-e53244e71082a392.parquet",
        "69fb1510e9990f8d0f62ab8f504ae3306130aec85c5f06fa772bce9d2db463a4", 155300860),
    "rog/validation-00000-of-00001-6ee6adc5b154643a.parquet": (
        HF + "validation-00000-of-00001-6ee6adc5b154643a.parquet",
        "8355b852e5aae75197d4220031f88a6b7627dfd457383edf29c7f9b9a92a272d", 24340437),
    "rog/test-00000-of-00002-9ee8d68f7d951e1f.parquet": (
        HF + "test-00000-of-00002-9ee8d68f7d951e1f.parquet",
        "3e756abdcdf944e72fb7408ff670ff33d030c1a81413338478904e8b07c64235", 90854720),
    "rog/test-00001-of-00002-773a7b8213e159f5.parquet": (
        HF + "test-00001-of-00002-773a7b8213e159f5.parquet",
        "0f7b81b759a0c0a7e675f567b7b378687c849d5aeb75cd50409de80b1c5b0a83", 93236638),
}
USER_AGENT = "RAG-course-project-data-fetch/1.0"


def ok(path: Path, sha: str, size: int) -> bool:
    return path.exists() and path.stat().st_size == size and sha256_file(path) == sha


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(NAMES_DIR))
    ap.add_argument("--copy-from", default=None,
                    help="directory holding entities_names.json and the RoG *.parquet shards (searched recursively)")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    out = ensure_dir(args.out)
    ensure_dir(out / "rog")
    local = {}
    if args.copy_from:
        for p in Path(args.copy_from).rglob("*"):
            if p.is_file():
                local.setdefault(p.name, p)
    bad = 0
    lines = []
    for rel, (url, sha, size) in FILES.items():
        dest = out / rel
        if ok(dest, sha, size):
            log(f"[ok]       {rel}")
        elif args.verify_only:
            log(f"[MISSING]  {rel}")
            bad += 1
            continue
        else:
            src = local.get(Path(rel).name)
            if src is not None:
                log(f"[copy]     {rel} <- {src}")
                shutil.copyfile(src, dest)
            else:
                log(f"[download] {rel}")
                download(url, dest)
            if not ok(dest, sha, size):
                log(f"[BAD HASH] {rel}")
                bad += 1
                continue
        lines.append(f"{sha}  {rel}")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if bad:
        log(f"{bad} file(s) missing or corrupt")
        return 1
    log(f"all {len(FILES)} files verified in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
