"""Format dataset.jsonl into a human-readable text file for qualitative inspection."""

import json
import os
import sys


def format_output(input_dir: str) -> None:
    jsonl_path = os.path.join(input_dir, "dataset.jsonl")
    out_path = os.path.join(input_dir, "readable_output.md")

    docs = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    out_lines = []
    for i, doc in enumerate(docs):
        meta = doc["metadata"]
        out_lines.append("=" * 80)
        out_lines.append(f"DOCUMENT {i + 1}: {meta['doc_id']}")
        out_lines.append(f"  Difficulty: {meta['difficulty']}")
        out_lines.append(f"  Style: {meta['style']}")
        out_lines.append(f"  Fachabteilung: {meta['fachabteilung']}")
        out_lines.append(f"  Sections: {meta['sections']}")
        out_lines.append(
            f"  Spans: {len(doc['spans'])}  Negatives: {len(doc['negatives'])}"
        )
        out_lines.append(f"  Flagged: {doc['qc']['flagged']}")
        if doc["qc"].get("reasons"):
            out_lines.append(f"  QC reasons: {doc['qc']['reasons']}")
        if meta.get("adversarial_mode"):
            out_lines.append(f"  Adversarial mode: {meta['adversarial_mode']}")
        out_lines.append("-" * 80)
        out_lines.append("")

        text = doc["text"]
        out_lines.append(text)
        out_lines.append("")

        out_lines.append("--- PII SPANS ---")
        for sp in doc["spans"]:
            surface = text[sp["start"] : sp["end"]]
            form = sp.get("form") or ""
            out_lines.append(
                f"  [{sp['start']}:{sp['end']}] {sp['label']:<20} "
                f"{form:<16} entity={sp['entity_id']:<12} "
                f'"{surface}"'
            )

        out_lines.append("")
        out_lines.append("--- NEGATIVE SPANS ---")
        for neg in doc["negatives"]:
            surface = text[neg["start"] : neg["end"]]
            out_lines.append(
                f'  [{neg["start"]}:{neg["end"]}] {neg["label"]:<20} "{surface}"'
            )

        out_lines.append("")
        out_lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))

    print(f"Wrote {len(docs)} documents to {out_path}")
    print(f"Total size: {os.path.getsize(out_path):,} bytes")


if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "output_test_p1"
    format_output(input_dir)
