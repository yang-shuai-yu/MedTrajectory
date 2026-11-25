"""Fetch a pinned trained MedTrajectory package, then run it entirely offline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from urllib.request import Request, urlopen

sys.dont_write_bytecode = True
SKILL = Path(__file__).resolve().parents[1]
LOCK = json.loads((SKILL / "references/upstream-lock.json").read_text(encoding="utf-8"))


def target(root, relative):
    rel = PurePosixPath(relative)
    if not relative or rel.is_absolute() or ".." in rel.parts or "\\" in relative or ":" in relative:
        raise ValueError("Expected a safe relative path")
    path = root.joinpath(*rel.parts)
    for node in (path, *path.parents):
        if node.is_symlink() or (hasattr(node, "is_junction") and node.is_junction()):
            raise ValueError("Linked paths are not supported in the managed workspace")
        if node == root:
            break
    path.resolve().relative_to(root.resolve())
    return path


def workspace(value):
    raw = Path(value).expanduser().absolute()
    if raw.is_symlink() or (hasattr(raw, "is_junction") and raw.is_junction()):
        raise ValueError("Workspace cannot be a symlink or junction")
    root = raw.resolve()
    if root == Path(root.anchor) or root == SKILL or SKILL in root.parents:
        raise ValueError("Choose an independent workspace directory")
    marker = target(root, ".medtrajectory-workspace.json")
    identity = {"repository": LOCK["repository"], "commit": LOCK["commit"]}
    if root.exists() and not marker.exists() and any(root.iterdir()):
        raise ValueError("Use an empty/new directory; existing unmarked directories are read-only")
    root.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != identity:
            raise ValueError("Workspace belongs to another version")
    else:
        with marker.open("x", encoding="utf-8") as stream:
            json.dump(identity, stream, indent=2)
    return root


def runtime(root):
    return target(root, "upstream/" + LOCK["commit"])


def verify(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        raise ValueError(f"Missing or wrong-sized file: {entry['path']}; run fetch first")
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError(f"Checksum mismatch: {entry['path']}")


def verify_all(root):
    for entry in LOCK["files"]:
        verify(target(runtime(root), entry["path"]), entry)


def fetch(root):
    downloaded = reused = 0
    for entry in LOCK["files"]:
        path = target(runtime(root), entry["path"])
        if path.exists():
            verify(path, entry)
            reused += 1
            continue
        if not 0 <= entry["size"] <= 16 * 1024 * 1024:
            raise ValueError("Unexpected file size in lock")
        url = f"https://raw.githubusercontent.com/{LOCK['repository']}/{LOCK['commit']}/{entry['path']}"
        with urlopen(Request(url, headers={"User-Agent": "MedTrajectorySkill/0.2.0"}), timeout=45) as response:
            if response.url != url:
                raise ValueError("Unexpected download redirect")
            content = response.read(entry["size"] + 1)
        if len(content) != entry["size"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"Download failed size/SHA-256 verification: {entry['path']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
        downloaded += 1
    return {"commit": LOCK["commit"], "downloaded": downloaded, "reused": reused,
            "runtime_directory": str(runtime(root)),
            "requirements": str(runtime(root) / "inference/requirements.txt"),
            "example_input": str(runtime(root) / "inference/examples/synthetic_history.json"),
            "trained_checkpoint_available": True}


def infer(root, input_path, output_name):
    verify_all(root)
    output = target(root, "outputs/" + output_name)
    if output.suffix != ".json" or output.exists():
        raise ValueError("Choose a new .json output name")
    command = [sys.executable, "-B", str(runtime(root) / "inference/predict.py"),
               "--input", str(Path(input_path).resolve()), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise ValueError(result.stderr.strip() or "Inference process failed")
    return {**json.loads(result.stdout), "source_commit": LOCK["commit"], "offline_inference": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("catalog")
    for name in ("fetch", "infer", "demo"):
        command = sub.add_parser(name)
        command.add_argument("--workspace", required=True)
        if name != "fetch":
            command.add_argument("--output", required=True, help="New JSON name under workspace/outputs")
        if name == "infer":
            command.add_argument("--input", required=True)
    args = parser.parse_args()
    if args.command == "catalog":
        result = {"repository": LOCK["repository"], "commit": LOCK["commit"],
                  "model_id": LOCK["model_id"], "trained_checkpoint_available": True,
                  "download_bytes": sum(e["size"] for e in LOCK["files"]),
                  "task": "multitype_trajectory_generation", "input_schema_version": 1}
    else:
        root = workspace(args.workspace)
        if args.command == "fetch":
            result = fetch(root)
        else:
            source = runtime(root) / "inference/examples/synthetic_history.json" if args.command == "demo" else args.input
            result = infer(root, source, args.output)
            result["synthetic_input"] = args.command == "demo"
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
