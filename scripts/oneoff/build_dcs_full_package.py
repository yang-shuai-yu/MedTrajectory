from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
PACKAGE_DIR = DIST / "MedTrajectory_DCS_Cloud_FULL"
ZIP_PATH = DIST / "MedTrajectory_DCS_Cloud_FULL.zip"

INCLUDE = [
    "README.md",
    "CODEX.md",
    "DCS_CLOUD_UPLOAD_README.md",
    "cpu_demo.py",
    "cpu_ckpt_pipeline.py",
    "requirements_cpu.txt",
    "requirements_dcs_cpu_torch.txt",
    "src",
    "scripts",
    "demo",
    "docs",
    "results",
]

EXCLUDE_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "dist",
    "outputs",
    "lo_manual_topk",
    "lo_profile_topk",
    "render_check_bert_rope",
    "rendered_report_qa",
    "report_render",
    "report_render_attrition",
    "report_render_expanded_20260702",
    "report_render_expanded_20260702_direct",
    "report_render_expanded_20260702_pdfium",
    "report_render_expanded_20260702_pdfpng",
    "report_render_frozen",
    "report_render_p0p1_20260701",
    "report_render_p0p1_20260701_final",
    "report_render_rollout",
    "report_render_topk_20260702",
    "report_render_topk_20260702_manual",
    "report_render_updated_20260629",
    "report_render_word",
}

EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".tmp"}


def should_skip(path: Path) -> bool:
    if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
        return True
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    return False


def copy_path(src: Path, dst: Path) -> None:
    if should_skip(src):
        return
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    for child in src.rglob("*"):
        if should_skip(child):
            continue
        if child.is_file():
            rel = child.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)


def main() -> int:
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    DIST.mkdir(parents=True, exist_ok=True)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

    missing = []
    for item in INCLUDE:
        src = ROOT / item
        if not src.exists():
            missing.append(item)
            continue
        copy_path(src, PACKAGE_DIR / item)

    if missing:
        print("Warning: missing package items:")
        for item in missing:
            print(f"  - {item}")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file in PACKAGE_DIR.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(DIST))

    total_files = sum(1 for p in PACKAGE_DIR.rglob("*") if p.is_file())
    total_size = sum(p.stat().st_size for p in PACKAGE_DIR.rglob("*") if p.is_file())
    print(f"Package directory: {PACKAGE_DIR}")
    print(f"Zip: {ZIP_PATH}")
    print(f"Files: {total_files}")
    print(f"Uncompressed size MB: {total_size / (1024 * 1024):.2f}")
    print(f"Zip size MB: {ZIP_PATH.stat().st_size / (1024 * 1024):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
