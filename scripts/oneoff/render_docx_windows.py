from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pdf2image import convert_from_path, pdfinfo_from_path


def prepend_bundled_bin_to_path() -> None:
    python_root = Path(sys.executable).resolve().parent.parent
    candidates = []
    if python_root.name == "python":
        dep_root = python_root.parent
        if dep_root.name == "dependencies":
            candidates.append(dep_root / "bin")
    for path in candidates:
        if path.exists():
            os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")


def find_soffice() -> Path:
    env_path = os.environ.get("SOFFICE", "").strip()
    if env_path and Path(env_path).exists():
        return Path(env_path)
    candidates = [
        Path(r"C:\Program Files\LibreOffice\program\soffice.com"),
        Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
        Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.com"),
        Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"),
    ]
    for path in candidates:
        if path.exists():
            return path
    found = shutil.which("soffice.com") or shutil.which("soffice")
    if found:
        return Path(found)
    raise FileNotFoundError("LibreOffice soffice executable was not found.")


def convert_docx_to_pdf(input_path: Path, output_dir: Path, soffice: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{input_path.stem}.pdf"
    if pdf_path.exists():
        pdf_path.unlink()
    cmd = [
        str(soffice),
        "--headless",
        "--norestore",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir),
        str(input_path),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=180)
    if proc.returncode != 0 or not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise RuntimeError(
            "LibreOffice PDF conversion failed.\n"
            f"returncode={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
    return pdf_path


def render_pdf_to_pngs(pdf_path: Path, output_dir: Path, dpi: int) -> list[Path]:
    for old in output_dir.glob("page-*.png"):
        old.unlink()
    info = pdfinfo_from_path(str(pdf_path))
    print(f"PDF pages: {info.get('Pages')}")
    raw_paths = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        fmt="png",
        thread_count=4,
        output_folder=str(output_dir),
        paths_only=True,
        output_file="page",
    )
    final_paths = []
    for raw in raw_paths:
        src = Path(raw)
        page_num = int(src.stem.split("-")[-1])
        dst = output_dir / f"page-{page_num}.png"
        if dst.exists():
            dst.unlink()
        src.rename(dst)
        final_paths.append(dst)
    return sorted(final_paths, key=lambda p: int(p.stem.split("-")[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Render DOCX to PDF and PNG on Windows via soffice.com.")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    input_path = args.input_path.resolve()
    output_dir = args.output_dir.resolve()
    prepend_bundled_bin_to_path()
    soffice = find_soffice()
    print(f"Using soffice: {soffice}")
    pdf_path = convert_docx_to_pdf(input_path, output_dir, soffice)
    print(f"PDF: {pdf_path}")
    pngs = render_pdf_to_pngs(pdf_path, output_dir, args.dpi)
    print(f"Rendered PNG pages: {len(pngs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
