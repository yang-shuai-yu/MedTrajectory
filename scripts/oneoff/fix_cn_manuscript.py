# -*- coding: utf-8 -*-
"""Apply the corresponding numeric/text fixes to Manuscript_cn_v1.docx.

Preserves images, tables, and formatting; only replaces paragraph text.
Equation changes (attention formula, training objective) and the added NLL
sentence are NOT done here (they are OMML math / insertions) and need a manual
Word pass.
"""
from docx import Document

PATH = r"E:\class\Doctor\term4\Comp1\MedTrajectory_AI_Competition\docs\articles\Manuscript_cn_v1.docx"

REPLACEMENTS = [
    # §5.4 locked-test risk numbers (replace validation numbers)
    ("0.733156", "0.745"),
    ("0.733428", "0.750"),
    ("0.731548", "0.743"),
    ("+0.000272（95% CI −0.003241--0.003795）", "+0.0047"),
    ("+0.000272", "+0.0047"),
    # §5.1 P0 baseline rename + remove CI
    ("0.1270（95% CI 0.1232--0.1309）", "0.127"),
    ("diagnosis-only 模型", "diagnosis-plus-death（Delphi 兼容）基线"),
    # §5.2 remove CIs
    ("0.01385（95% CI 0.01065--0.01706）", "0.01385"),
    ("0.03215（0.02077--0.04394）", "0.03215"),
    ("降低 5.01 天（95% CI 3.44--6.57）", "降低 5.01 天"),
    ("0.899（0.787--1.007）", "0.899"),
    # §5.6 remove CIs + death wording
    ("0.0305（95% CI −0.0390 至 −0.0227）", "0.0305"),
    ("11.35（95% CI 10.96--11.76）", "11.35"),
    ("0.0151（95% CI 0.0117--0.0189）", "0.0151"),
    ("0.0360（0.0204--0.0511）", "0.0360"),
    ("真实死亡率", "观测死亡率"),
    ("0.0970（95% CI 0.0367--0.1552）", "0.097"),
    # references
    ("PMC12657216", "PMC12589094"),
    ("Ishihara S", "Sawano S"),
]


def replace_in_paragraph(paragraph, counts):
    runs = paragraph.runs
    if not runs:
        return
    text = paragraph.text
    new_text = text
    for old, new in REPLACEMENTS:
        if old in new_text:
            new_text = new_text.replace(old, new)
            counts[old] = counts.get(old, 0) + new_text.count(old)
    if new_text != text:
        # rewrite paragraph text into the first run, clear the rest
        runs[0].text = new_text
        for run in runs[1:]:
            run.text = ""


def main():
    doc = Document(PATH)
    counts = {}
    for paragraph in doc.paragraphs:
        replace_in_paragraph(paragraph, counts)
    # tables too (references may be in tables; body tables have numbers)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    replace_in_paragraph(paragraph, counts)
    doc.save(PATH)
    print("saved:", PATH)
    for old, n in sorted(counts.items()):
        print(f"  {n:>3}x  {old}")


if __name__ == "__main__":
    main()
