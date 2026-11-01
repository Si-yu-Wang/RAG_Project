"""Reorganise docs/ by function and fix every reference in one pass.

Groups:
    contract/    authoritative schema + validation reports
    dataset/     corpus profiling, cleaning, alignment, per-dataset reports
    experiments/ grid results, per-config stats, probes
    design/      plans, reviews, comparability notes

Moving files without rewriting the links that point at them is how a docs folder rots, so the
same script that moves everything also rewrites every markdown reference and every script's
default output path. Idempotent: running it twice is a no-op.
"""
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(r"D:\Users\NTU\6405\project\RAG_Project")

# ---------------------------------------------------------------- 1. moves
MOVES = {
    # contract/
    "docs/data_contract.md": "docs/contract/data_contract.md",
    "docs/contract_validation.md": "docs/contract/contract_validation.md",
    "docs/contract_validation_hard.md": "docs/contract/contract_validation_hard.md",
    "docs/contract_validation_pubmedqa.md": "docs/contract/contract_validation_pubmedqa.md",
    # dataset/
    "docs/dataset_profile.md": "docs/dataset/dataset_profile.md",
    "docs/profile_stats.json": "docs/dataset/profile_stats.json",
    "docs/figures": "docs/dataset/figures",
    "docs/cleaning_report.md": "docs/dataset/cleaning_report.md",
    "docs/evidence_alignment_report.md": "docs/dataset/evidence_alignment_report.md",
    "docs/dataset_consistency.md": "docs/dataset/dataset_consistency.md",
    "docs/dataset_consistency.md.json": "docs/dataset/dataset_consistency.md.json",
    "docs/pubmedqa_report.md": "docs/dataset/pubmedqa_report.md",
    # experiments/
    "docs/chunk_grid_results.md": "docs/experiments/chunk_grid_results.md",
    "docs/chunk_grid_hard.md": "docs/experiments/chunk_grid_hard.md",
    "docs/chunking_stats_size512_ov102_soft.md":
        "docs/experiments/chunking_stats_size512_ov102_soft.md",
    "docs/chunking_stats_size512_ov102_hard.md":
        "docs/experiments/chunking_stats_size512_ov102_hard.md",
    "docs/chunking_stats_pubmedqa_512_ov102.md":
        "docs/experiments/chunking_stats_pubmedqa_512_ov102.md",
    "docs/chunking_stats_pubmedqa_128_ov26.md":
        "docs/experiments/chunking_stats_pubmedqa_128_ov26.md",
    "docs/grid_results.csv": "docs/experiments/grid_results.csv",
    "docs/grid_results_hard.csv": "docs/experiments/grid_results_hard.csv",
    "docs/citation_graph_probe.md": "docs/experiments/citation_graph_probe.md",
    # design/
    "docs/Dataset_Chunking_技术清单.md": "docs/design/Dataset_Chunking_技术清单.md",
    "docs/M1_实施计划_v2.md": "docs/design/M1_实施计划_v2.md",
    "docs/图检索实验设计评审.md": "docs/design/图检索实验设计评审.md",
    "docs/跨数据集可比性说明.md": "docs/design/跨数据集可比性说明.md",
}

# ------------------------------------------------- 2. reference rewrites, per file
# Keys are repo-relative paths; values are (old, new) pairs applied in order.
D = "docs/"
R = {
    # from repo root (README)
    "README.md": [
        (f"{D}data_contract.md", f"{D}contract/data_contract.md"),
        (f"{D}citation_graph_probe.md", f"{D}experiments/citation_graph_probe.md"),
        (f"{D}chunk_grid_results.md", f"{D}experiments/chunk_grid_results.md"),
        (f"{D}chunk_grid_hard.md", f"{D}experiments/chunk_grid_hard.md"),
        (f"{D}contract_validation", f"{D}contract/contract_validation"),
        (f"{D}dataset_profile.md", f"{D}dataset/dataset_profile.md"),
        (f"{D}dataset_consistency.md", f"{D}dataset/dataset_consistency.md"),
        (f"{D}pubmedqa_report.md", f"{D}dataset/pubmedqa_report.md"),
        (f"{D}跨数据集可比性说明.md", f"{D}design/跨数据集可比性说明.md"),
    ],
    # from docs/ root (PROGRESS)
    "docs/PROGRESS.md": [
        ("docs/data_contract.md", "contract/data_contract.md"),
        ("docs/contract_validation", "contract/contract_validation"),
        ("docs/cleaning_report.md", "dataset/cleaning_report.md"),
        ("docs/dataset_profile.md", "dataset/dataset_profile.md"),
        ("docs/evidence_alignment_report.md", "dataset/evidence_alignment_report.md"),
        ("docs/dataset_consistency.md", "dataset/dataset_consistency.md"),
        ("docs/pubmedqa_report.md", "dataset/pubmedqa_report.md"),
        ("docs/citation_graph_probe.md", "experiments/citation_graph_probe.md"),
        ("docs/chunk_grid", "experiments/chunk_grid"),
        ("docs/grid_results", "experiments/grid_results"),
        ("docs/figures", "dataset/figures"),
        ("(Dataset_Chunking_技术清单.md)", "(design/Dataset_Chunking_技术清单.md)"),
        ("(M1_实施计划_v2.md)", "(design/M1_实施计划_v2.md)"),
        ("`Dataset_Chunking_技术清单.md`", "`design/Dataset_Chunking_技术清单.md`"),
        ("`M1_实施计划_v2.md`", "`design/M1_实施计划_v2.md`"),
    ],
    # from docs/contract/
    "docs/contract/data_contract.md": [
        ("`PROGRESS.md`", "`../PROGRESS.md`"),
        ("`chunk_grid_results.md`", "`../experiments/chunk_grid_results.md`"),
        ("`cleaning_report.md`", "`../dataset/cleaning_report.md`"),
    ],
    # from docs/dataset/
    "docs/dataset/pubmedqa_report.md": [
        ("docs/跨数据集可比性说明.md", "../design/跨数据集可比性说明.md"),
        ("docs/contract_validation_pubmedqa.md", "../contract/contract_validation_pubmedqa.md"),
    ],
    # from docs/design/
    "docs/design/Dataset_Chunking_技术清单.md": [
        ("(dataset_profile.md)", "(../dataset/dataset_profile.md)"),
        ("(cleaning_report.md)", "(../dataset/cleaning_report.md)"),
        ("(PROGRESS.md)", "(../PROGRESS.md)"),
        ("docs/dataset_profile.md", "../dataset/dataset_profile.md"),
        ("docs/data_contract.md", "../contract/data_contract.md"),
        ("docs/evidence_alignment_report.md", "../dataset/evidence_alignment_report.md"),
        ("docs/chunking_experiments.md", "../experiments/chunking_experiments.md"),
    ],
    "docs/design/M1_实施计划_v2.md": [
        ("docs/dataset_profile.md", "../dataset/dataset_profile.md"),
        ("docs/cleaning_report.md", "../dataset/cleaning_report.md"),
        ("docs/evidence_alignment_report.md", "../dataset/evidence_alignment_report.md"),
        ("docs/chunk_grid_results.md", "../experiments/chunk_grid_results.md"),
        ("docs/pubmedqa_smoke.md", "../experiments/pubmedqa_smoke.md"),
        ("docs/data_contract.md", "../contract/data_contract.md"),
        ("`citation_graph_probe.md`", "`../experiments/citation_graph_probe.md`"),
    ],
    "docs/design/图检索实验设计评审.md": [
        ("`citation_graph_probe.md`", "`../experiments/citation_graph_probe.md`"),
        ("`evidence_alignment_report.md`", "`../dataset/evidence_alignment_report.md`"),
    ],
}

# ------------------------------------------------- 3. script default output paths
SCRIPT_PATHS = {
    "scripts/00_profile_dataset.py": [('default="docs"', 'default="docs/dataset"'),
                                      ("docs/dataset_profile.md", "docs/dataset/dataset_profile.md")],
    "scripts/01_prepare_corpus.py": [('default="docs"', 'default="docs/dataset"')],
    "scripts/02_align_evidence.py": [('default="docs"', 'default="docs/dataset"')],
    "scripts/03_build_chunks.py": [('default="docs"', 'default="docs/experiments"'),
                                   ("docs/chunking_stats_<name>.md",
                                    "docs/experiments/chunking_stats_<name>.md")],
    "scripts/04_chunk_grid.py": [('default="docs"', 'default="docs/experiments"')],
    "scripts/05_validate_contract.py": [
        ('default="docs/contract_validation.md"', 'default="docs/contract/contract_validation.md"')],
    "scripts/dev/probe_citation_graph.py": [
        ('default="docs/citation_graph_probe.md"',
         'default="docs/experiments/citation_graph_probe.md"')],
    "scripts/dev/verify_against_paper.py": [
        ('default="docs/dataset_consistency.md"',
         'default="docs/dataset/dataset_consistency.md"')],
    "scripts/10_prepare_pubmedqa.py": [('default="docs"', 'default="docs/dataset"')],
}


def run_git(*args: str) -> None:
    subprocess.run(["git", *args], cwd=str(ROOT), check=True,
                   capture_output=True)


def main() -> None:
    moved = skipped = 0
    for src, dst in MOVES.items():
        s, d = ROOT / src, ROOT / dst
        if not s.exists():
            skipped += 1
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.is_dir():
            shutil.move(str(s), str(d))
        else:
            shutil.move(str(s), str(d))
        moved += 1
    print(f"moved={moved} skipped(already gone)={skipped}")

    rewritten = 0
    for rel, pairs in list(R.items()) + list(SCRIPT_PATHS.items()):
        p = ROOT / rel
        if not p.exists():
            print(f"!! missing (skip rewrites): {rel}")
            continue
        text = p.read_text(encoding="utf-8")
        orig = text
        for old, new in pairs:
            text = text.replace(old, new)
        if text != orig:
            p.write_text(text, encoding="utf-8", newline="\n")
            rewritten += 1
    print(f"files_with_rewrites={rewritten}")

    # stage everything so git records the renames explicitly
    run_git("add", "-A")
    print("staged; use git status to review renames")


if __name__ == "__main__":
    main()
