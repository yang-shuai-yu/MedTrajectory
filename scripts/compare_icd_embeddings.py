"""Compare ICD-10 encoding methods by intrinsic "semantic separation" quality.

Encodes ICD-10 level-3 codes with several methods, PCA-reduces to a common
dimension, clusters with k-means, and scores how well the clusters recover the
ICD-10 chapter / block structure (the gold-standard grouping labels). This is a
REPRESENTATION-QUALITY study only; it does NOT train or ablate the downstream
trajectory model.

Methods:
  hierarchy   - deterministic one-hot over ICD-10 chapter (22 chapters)
  gram        - deterministic ontology-aware block one-hot (263 blocks; Chapter->Block->Code)
  qwen        - Qwen text-embedding-v4 via DashScope compatible-mode API (1024d)
  glm         - GLM embedding-3 via ZhipuAI API (out of scope; needs ZHIPUAI_API_KEY)
  pubmedbert  - PubMedBERT / BiomedCLIP text-encoder mean-pooled embedding

Evaluation labels (gold standard):
  chapter     - ICD-10 chapter name (from the Delphi labels CSV)
  block       - ICD-10 block/group (from icd102019syst_groups.txt, via --icd-dir)

Usage:
  python scripts/compare_icd_embeddings.py --csv delphi_labels_chapters_colours_icd.csv \
      --icd-dir path/to/icd10_ontology --methods hierarchy gram qwen --out outputs/icd_embedding_compare
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

DEFAULT_DELPHI_CSV_URL = (
    "https://raw.githubusercontent.com/gerstung-lab/Delphi/main/"
    "delphi_labels_chapters_colours_icd.csv"
)


# --------------------------------------------------------------------------- #
# 1. Load ICD-10 code -> description + chapter from the Delphi label CSV       #
# --------------------------------------------------------------------------- #
def sniff_csv_dicts(text: str) -> List[Dict[str, str]]:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.excel
    return [dict(row) for row in csv.DictReader(text.splitlines(), dialect=dialect)]


def load_delphi_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Delphi CSV not found: {path}")
    rows = sniff_csv_dicts(path.read_text(encoding="utf-8-sig"))
    records: Dict[str, Dict[str, str]] = {}
    for row in rows:
        code, description, chapter = infer_code_and_description(row)
        if not code:
            continue
        rec = records.setdefault(code, {"code": code, "description": description, "chapter": chapter})
        if chapter and not rec["chapter"]:
            rec["chapter"] = chapter
        if description and len(description) > len(rec["description"]):
            rec["description"] = description
    return list(records.values())


def infer_code_and_description(row: Dict[str, str]):
    """Parse a delphi_labels_chapters_colours_icd.csv row -> (code, description, chapter).

    `name` is "<ICD-10 3-char code> <English description>" (e.g. "A00 Cholera");
    special tokens (Padding / Female / BMI low / Death) have no leading code.
    """
    lowered = {str(k).strip().lower(): str(v).strip() for k, v in row.items()}
    name = lowered.get("name", "")
    if len(name) < 4 or not name[0].isalpha() or not name[1].isdigit():
        return None, "", ""
    code = name[:3].upper()
    description = name[3:].lstrip(" -.")
    chapter = lowered.get("icd-10 chapter", "") or lowered.get("icd-10 chapter (short)", "")
    if not chapter:
        chapter = lowered.get("chapter", "")
    return code, description, chapter


def provider_text(rec: Dict[str, str]) -> str:
    lines = [
        f"ICD-10 level-3 code: {rec['code']}",
        f"Disease description: {rec['description']}",
    ]
    if rec.get("chapter"):
        lines.append(f"ICD chapter: {rec['chapter']}")
    lines.append("Task hint: represent the medical semantics of this ICD-10 token for downstream trajectory modelling.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 2. Encoders: code -> fixed-dim vector                                        #
# --------------------------------------------------------------------------- #
def encode_hierarchy(records: List[Dict[str, str]]) -> Dict[str, np.ndarray]:
    """Deterministic baseline: one-hot over the ICD-10 chapters."""
    chapters = sorted({r["chapter"] for r in records if r.get("chapter")})
    idx = {c: i for i, c in enumerate(chapters)}
    out: Dict[str, np.ndarray] = {}
    for r in records:
        vec = np.zeros(len(chapters), dtype=np.float32)
        if r.get("chapter") in idx:
            vec[idx[r["chapter"]]] = 1.0
        out[r["code"]] = vec
    return out


def _codes_in_range(start: str, end: str) -> List[str]:
    letter = start[0]
    return [f"{letter}{n:02d}" for n in range(int(start[1:]), int(end[1:]) + 1)]


def load_icd_ontology(icd_dir: Path):
    chapters = {}
    for line in (icd_dir / "icd102019syst_chapters.txt").read_text(encoding="utf-8").splitlines():
        num, name = line.split(";", 1)
        chapters[int(num)] = name.strip()
    groups = []
    for line in (icd_dir / "icd102019syst_groups.txt").read_text(encoding="utf-8").splitlines():
        start, end, ch_num, name = line.split(";", 3)
        groups.append((start.strip(), end.strip(), int(ch_num), name.strip()))
    code_to_block: Dict[str, int] = {}
    block_names = []
    for idx, (start, end, _ch, name) in enumerate(groups):
        block_names.append(name)
        for code in _codes_in_range(start, end):
            code_to_block[code] = idx
    return chapters, groups, code_to_block, block_names


def encode_gram(records: List[Dict[str, str]], icd_dir: Path) -> Dict[str, np.ndarray]:
    """Deterministic ontology-aware encoding (GRAM-style, block level).

    Uses the ICD-10 Chapter -> Block(group) -> 3-char code hierarchy: each code is
    a one-hot over its block (the intermediate ancestor), which is the ontology
    structure GRAM leverages. Learned attention is intentionally omitted (this is
    an intrinsic, no-training comparison).
    """
    _, _, code_to_block, block_names = load_icd_ontology(Path(icd_dir))
    n_blocks = len(block_names)
    out: Dict[str, np.ndarray] = {}
    for r in records:
        vec = np.zeros(n_blocks, dtype=np.float32)
        idx = code_to_block.get(r["code"])
        if idx is not None:
            vec[idx] = 1.0
        out[r["code"]] = vec
    return out


def load_phecode_map(phecode_csv: Path) -> Dict[str, str]:
    """Build ICD-10 3-char code -> Phecode parent group (integer prefix).

    Reads a CSV with columns (ICD_id, Phecode, ...) such as the PheWAS v1.2
    ICD-10-CM map. Each 3-char ICD code maps to its Phecode's 3-digit parent
    (e.g. I10 -> 401, E11 -> 250), which is the clinically-meaningful grouping.
    """
    import pandas as pd
    df = pd.read_csv(phecode_csv, dtype=str)
    code_to_phe: Dict[str, str] = {}
    for icd, phe in zip(df["ICD_id"], df["Phecode"]):
        icd = str(icd).strip().upper()
        phe = str(phe).strip()
        if len(icd) >= 3 and icd[0].isalpha() and icd[1].isdigit():
            code_to_phe.setdefault(icd[:3], phe.split(".")[0])
    return code_to_phe


def _post_embeddings(texts: Sequence[str], url: str, api_key: str, model: str, dimensions: int, batch_size: int = 4) -> np.ndarray:
    all_out = []
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start:start + batch_size])
        payload = json.dumps({"model": model, "input": chunk, "dimensions": dimensions, "encoding_format": "float"}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        ordered = sorted(data["data"], key=lambda item: int(item.get("index", 0)))
        all_out.extend([item["embedding"] for item in ordered])
    return np.asarray(all_out, dtype=np.float32)


def encode_qwen(records: List[Dict[str, str]], api_key: str) -> Dict[str, np.ndarray]:
    texts = [provider_text(r) for r in records]
    mat = _post_embeddings(texts, "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
                           api_key, "text-embedding-v4", 1024)
    return {r["code"]: mat[i] for i, r in enumerate(records)}


def encode_glm(records: List[Dict[str, str]], api_key: str) -> Dict[str, np.ndarray]:
    texts = [provider_text(r) for r in records]
    mat = _post_embeddings(texts, "https://open.bigmodel.cn/api/paas/v4/embeddings",
                           api_key, "embedding-3", 1024)
    return {r["code"]: mat[i] for i, r in enumerate(records)}


def encode_pubmedbert(records: List[Dict[str, str]], model_name: str) -> Dict[str, np.ndarray]:
    """Text-encoder mean-pooled embedding from a HF biomedical model.

    Works with a plain BERT (e.g. microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)
    or a CLIP-style model (e.g. microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224,
    whose text tower is PubMedBERT).
    """
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pip install transformers torch") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    out: Dict[str, np.ndarray] = {}
    for r in records:
        text = provider_text(r)  # same rich prompt as Qwen (code + description + chapter + hint)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state  # [1, T, D]
        elif hasattr(outputs, "text_model_output"):
            hidden = outputs.text_model_output.last_hidden_state
        else:
            raise RuntimeError(f"cannot locate a text hidden state for model {model_name}")
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1)  # mean pool
        out[r["code"]] = pooled[0].cpu().numpy().astype(np.float32)
    return out


ENCODERS = {
    "hierarchy": lambda recs, **kw: encode_hierarchy(recs),
    "gram": lambda recs, **kw: encode_gram(recs, kw["icd_dir"]),
    "qwen": lambda recs, **kw: encode_qwen(recs, kw["api_key"]),
    "glm": lambda recs, **kw: encode_glm(recs, kw["api_key"]),
    "pubmedbert": lambda recs, **kw: encode_pubmedbert(recs, kw["pubmedbert_model"]),
}


# --------------------------------------------------------------------------- #
# 3. PCA -> common dim                                                        #
# --------------------------------------------------------------------------- #
def pca_reduce(matrix: np.ndarray, dim: int) -> np.ndarray:
    matrix = matrix - matrix.mean(0, keepdims=True)
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    return (u[:, :dim] * s[:dim]).astype(np.float32)


# --------------------------------------------------------------------------- #
# 4. Evaluation: k-means + ARI/NMI/Silhouette vs labels                        #
# --------------------------------------------------------------------------- #
def evaluate(matrix: np.ndarray, labels: np.ndarray, n_clusters: int) -> Dict[str, float]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(matrix)
    pred = km.labels_
    return {
        "ari": float(adjusted_rand_score(labels, pred)),
        "nmi": float(normalized_mutual_info_score(labels, pred)),
        "silhouette": float(silhouette_score(matrix, pred)) if n_clusters > 1 else float("nan"),
    }


def umap_plot(matrix: np.ndarray, labels: np.ndarray, label_names: Sequence[str], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import umap
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pip install umap-learn matplotlib") from exc
    embedding = umap.UMAP(n_components=2, random_state=0).fit_transform(matrix)
    fig, ax = plt.subplots(figsize=(12, 9))
    for lab in np.unique(labels):
        m = labels == lab
        ax.scatter(embedding[m, 0], embedding[m, 1], s=6, label=str(label_names[int(lab)]), alpha=0.7)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. Main                                                                      #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, help="Local delphi_labels_chapters_colours_icd.csv")
    ap.add_argument("--csv-url", default=DEFAULT_DELPHI_CSV_URL, help="Fallback URL to download the CSV")
    ap.add_argument("--methods", nargs="+", default=["hierarchy"], choices=sorted(ENCODERS))
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("outputs/icd_embedding_compare"))
    ap.add_argument("--pubmedbert-model", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    ap.add_argument("--api-key", help="Qwen(DASHSCOPE) or GLM(ZHIPU) key; else read env")
    ap.add_argument("--icd-dir", type=Path, default=None, help="dir with icd102019syst_chapters.txt / _groups.txt")
    ap.add_argument("--phecode-csv", type=Path, default=None, help="ICD_id,Phecode,Phenotype CSV (PheWAS map)")
    args = ap.parse_args()

    csv_path = args.csv
    if csv_path is None or not csv_path.exists():
        csv_path = args.out / "delphi_labels_chapters_colours_icd.csv"
        if not csv_path.exists():
            print(f"Downloading {args.csv_url} ...")
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(args.csv_url, csv_path)

    records = load_delphi_csv(csv_path)
    records = [r for r in records if r.get("chapter")]
    if not records:
        raise RuntimeError("no ICD-10 records with chapter labels extracted from CSV")

    chapter_names = sorted({r["chapter"] for r in records})
    chapter_id = {c: i for i, c in enumerate(chapter_names)}
    codes = [r["code"] for r in records]
    gold = np.asarray([chapter_id[r["chapter"]] for r in records], dtype=np.int64)

    block_gold = None
    block_names = None
    if args.icd_dir is not None:
        _, _, code_to_block, block_names = load_icd_ontology(args.icd_dir)
        block_gold = np.asarray([code_to_block.get(c, -1) for c in codes], dtype=np.int64)

    phecode_gold = None
    if args.phecode_csv is not None:
        code_to_phe = load_phecode_map(args.phecode_csv)
        phe_labels = [code_to_phe.get(c, "") for c in codes]
        phecode_names = sorted({p for p in phe_labels if p})
        phecode_id = {p: i for i, p in enumerate(phecode_names)}
        phecode_gold = np.asarray([phecode_id[p] if p in phecode_id else -1 for p in phe_labels], dtype=np.int64)

    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("ZHIPUAI_API_KEY")
    results = []
    for method in args.methods:
        print(f"encoding {method} ({len(records)} codes) ...")
        embeddings = ENCODERS[method](
            records, api_key=api_key, pubmedbert_model=args.pubmedbert_model, icd_dir=args.icd_dir
        )
        matrix = np.stack([embeddings[c] for c in codes])   # native dim
        matrix64 = pca_reduce(matrix, args.dim)              # PCA -> 64d

        def _score(mat, prefix):
            out = {}
            m = evaluate(mat, gold, n_clusters=len(chapter_names))
            out[prefix + "chapter_ari"] = m["ari"]
            out[prefix + "chapter_nmi"] = m["nmi"]
            out[prefix + "chapter_silhouette"] = m["silhouette"]
            if block_gold is not None:
                valid = block_gold >= 0
                nbp = int(np.unique(block_gold[valid]).size)
                bm = evaluate(mat[valid], block_gold[valid], n_clusters=nbp)
                out[prefix + "block_ari"] = bm["ari"]
                out[prefix + "block_nmi"] = bm["nmi"]
                out["n_blocks"] = nbp
            if phecode_gold is not None:
                valid = phecode_gold >= 0
                nph = int(np.unique(phecode_gold[valid]).size)
                pm = evaluate(mat[valid], phecode_gold[valid], n_clusters=nph)
                out[prefix + "phecode_ari"] = pm["ari"]
                out[prefix + "phecode_nmi"] = pm["nmi"]
                out["n_phecodes"] = nph
            return out

        metrics = {"method": method, "n_codes": len(codes), "native_dim": int(matrix.shape[1])}
        metrics.update(_score(matrix, "native_"))
        metrics.update(_score(matrix64, ""))
        results.append(metrics)
        umap_plot(matrix64, gold, chapter_names, args.out / f"umap_{method}.png")
        np.save(args.out / f"emb_{method}_{args.dim}d.npy", matrix64)
        np.save(args.out / f"emb_{method}_native.npy", matrix)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "chapter_names.json").write_text(json.dumps(chapter_names, ensure_ascii=False, indent=2))
    if block_names is not None:
        (args.out / "block_names.json").write_text(json.dumps(block_names, ensure_ascii=False, indent=2))
    (args.out / "comparison.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))

    print(f"\n{'method':<12}{'dim':>6}{'chA64':>7}{'chAnat':>7}{'blkA64':>7}{'blkAnat':>7}{'pheA64':>7}{'pheAnat':>7}{'pheN64':>7}{'pheNnat':>7}")
    for r in results:
        print(f"{r['method']:<12}{r['native_dim']:>6}{r['chapter_ari']:>7.3f}{r['native_chapter_ari']:>7.3f}"
              f"{r.get('block_ari', float('nan')):>7.3f}{r.get('native_block_ari', float('nan')):>7.3f}"
              f"{r.get('phecode_ari', float('nan')):>7.3f}{r.get('native_phecode_ari', float('nan')):>7.3f}"
              f"{r.get('phecode_nmi', float('nan')):>7.3f}{r.get('native_phecode_nmi', float('nan')):>7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
