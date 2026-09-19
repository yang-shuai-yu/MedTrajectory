from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"


def venv_python() -> Path:
    if sys.platform.startswith("win"):
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(x) for x in cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def has_pip(py: Path) -> bool:
    try:
        subprocess.check_call([str(py), "-m", "pip", "--version"], cwd=ROOT)
        return True
    except Exception:
        return False


def create_venv() -> None:
    print(f"Creating .venv at {VENV_DIR}")
    try:
        venv.EnvBuilder(with_pip=True, clear=False).create(VENV_DIR)
    except Exception as exc:
        print(f"stdlib venv failed: {exc}")
        print("Trying virtualenv fallback via current Python/pip...")
        run([sys.executable, "-m", "pip", "install", "--user", "virtualenv"])
        run([sys.executable, "-m", "virtualenv", str(VENV_DIR)])


def main() -> int:
    parser = argparse.ArgumentParser(description="Create DCS Cloud CPU demo .venv.")
    parser.add_argument("--with-torch", action="store_true", help="Install CPU PyTorch for checkpoint smoke inference.")
    parser.add_argument("--force", action="store_true", help="Recreate .venv if it already exists.")
    args = parser.parse_args()

    if sys.version_info < (3, 10):
        raise SystemExit("Python >= 3.10 is required; Python 3.12 is recommended on DCS Cloud.")

    if VENV_DIR.exists() and args.force:
        shutil.rmtree(VENV_DIR)

    if not VENV_DIR.exists():
        create_venv()
    else:
        print(f"Using existing .venv at {VENV_DIR}")

    py = venv_python()
    if not py.exists():
        raise SystemExit(f"Virtualenv Python not found: {py}. Remove .venv and rerun this script.")

    if not has_pip(py):
        print("Existing .venv has no pip; it is likely a broken venv from missing python3.12-venv.")
        print("Recreating .venv via virtualenv fallback...")
        shutil.rmtree(VENV_DIR)
        run([sys.executable, "-m", "pip", "install", "--user", "virtualenv"])
        run([sys.executable, "-m", "virtualenv", str(VENV_DIR)])
        py = venv_python()

    run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    if args.with_torch:
        # Install NumPy from the normal Python package index first. Do not use
        # the PyTorch CPU wheel index for NumPy, otherwise NumPy may be missing.
        run([str(py), "-m", "pip", "install", "numpy>=1.26,<3"])
        run([str(py), "-m", "pip", "install", "torch>=2.7", "--index-url", "https://download.pytorch.org/whl/cpu"])
    else:
        print("Standard artifact demo needs no external packages.")

    print("\nRun artifact demo:")
    print(f"  {py} cpu_demo.py")
    if args.with_torch:
        print("\nRun checkpoint CPU smoke pipeline:")
        print(f"  {py} cpu_ckpt_pipeline.py")
    else:
        print("\nFor checkpoint CPU smoke pipeline, rerun:")
        print(f"  {sys.executable} scripts/setup_dcs_cpu_venv.py --with-torch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
