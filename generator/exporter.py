import json
import os
from collections import Counter
from typing import Any, Dict, List


class Exporter:
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def write_jsonl(self, rows: List[Dict[str, Any]], filename: str) -> str:
        target = os.path.join(self.output_dir, filename)
        with open(target, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return target

    def write_json(self, obj: Any, filename: str) -> str:
        target = os.path.join(self.output_dir, filename)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return target

    def export_review_queue(
        self, records: List[Dict[str, Any]], filename: str = "review_queue.jsonl"
    ) -> str:
        """Write every flagged document (with the full QC reasoning) for review.

        In ``auto`` mode a flagged document is *excluded* from the dataset, so
        this file is the authoritative record of every exclusion decision — both
        deterministic (leaks / negative-integrity / entity-collapse) and
        red-team findings. Must be called on the complete record set, before any
        auto-filtering drops the flagged docs.
        """
        flagged = [
            {
                "doc_id": r["doc_id"],
                "difficulty": r.get("metadata", {}).get("difficulty"),
                "style": r.get("metadata", {}).get("style"),
                "reasons": r["qc"]["reasons"],
                "leaks": r["qc"].get("leaks", []),
                "negative_findings": r["qc"].get("negative_findings", []),
                "missing_planned_persons": r["qc"].get("missing_planned_persons", []),
                "red_team": r["qc"].get("red_team"),
                "text": r["text"],
            }
            for r in records
            if r.get("qc", {}).get("flagged")
        ]
        return self.write_jsonl(flagged, filename)

    def export_failures(
        self, failures: List[Dict[str, Any]], filename: str = "failures.jsonl"
    ) -> str:
        """Persist every document that was DROPPED before producing a record.

        A drop is a hard failure in Stages 1-3 (LLM authoring, skeleton quality,
        render, or span-integrity) that prevented a valid gold record from being
        built at all — as opposed to a QC *exclusion* (a fully rendered doc that
        the deterministic checks / red team flagged, recorded in the review
        queue). Each entry carries ``doc_id``, ``seed``, ``stage``,
        ``error_type`` and the ``error`` message so the yield gap is fully
        accounted for. Always written (an empty file documents zero drops).
        """
        return self.write_jsonl(failures, filename)

    def export_provenance(
        self,
        records: List[Dict[str, Any]],
        failures: List[Dict[str, Any]],
        *,
        auto_mode: bool,
        filename: str = "provenance.json",
    ) -> tuple[str, Dict[str, Any]]:
        """Aggregate the full generation provenance / yield chain for the paper.

        Must be called on the COMPLETE successfully-generated record set (before
        any auto-filter drops flagged docs) plus the drop list from
        :meth:`export_failures`. Reports:

        * yield — attempted -> dropped -> generated -> excluded -> kept, w/ rates;
        * generation drops broken down by stage and error type;
        * QC exclusions broken down by mechanism, plus the deterministic-vs-red
          team *overlap* (caught by only one, or both);
        * findings broken down *by label type* (leaked PHI labels, negative-trap
          labels, red-team finding categories).

        The per-sample detail lives in ``failures.jsonl`` (drops) and
        ``review_queue.jsonl`` (exclusions); this file is the aggregate view.
        """
        attempted = len(records) + len(failures)
        flagged = [r for r in records if r.get("qc", {}).get("flagged")]
        kept = len(records) - len(flagged) if auto_mode else len(records)

        # --- generation drops ---
        drop_stage = Counter(f.get("stage", "unknown") for f in failures)
        drop_type = Counter(f.get("error_type", "unknown") for f in failures)
        drop_stage_error = Counter(
            (f.get("stage", "unknown"), f.get("error_type", "unknown"))
            for f in failures
        )

        # --- QC exclusion mechanisms + deterministic/red-team overlap ---
        mech = Counter()
        overlap = Counter()
        leak_labels: Counter = Counter()
        neg_labels: Counter = Counter()
        neg_reasons: Counter = Counter()
        rt_categories: Counter = Counter()
        rt_kinds: Counter = Counter()
        for r in flagged:
            qc = r.get("qc", {})
            leaks = qc.get("leaks") or []
            negs = qc.get("negative_findings") or []
            reasons = qc.get("reasons") or []
            rt = qc.get("red_team") or {}
            review_items = rt.get("review_items") or []
            rt_error = rt.get("status") == "error"
            entity_collapse = any(str(x).startswith("entity collapse") for x in reasons)

            if leaks:
                mech["deterministic_leak"] += 1
            if negs:
                mech["negative_integrity"] += 1
            if entity_collapse:
                mech["entity_collapse"] += 1
            if review_items:
                mech["redteam_findings"] += 1
            if rt_error:
                mech["redteam_call_error"] += 1

            det = bool(leaks) or bool(negs) or entity_collapse
            rt_flag = bool(review_items) or rt_error
            if det and rt_flag:
                overlap["both"] += 1
            elif det:
                overlap["deterministic_only"] += 1
            elif rt_flag:
                overlap["redteam_only"] += 1

            # by-label findings (aggregate over every finding, not per doc)
            for lk in leaks:
                leak_labels[lk.get("label", "UNKNOWN")] += 1
            for nf in negs:
                neg_labels[nf.get("label", "UNKNOWN")] += 1
                neg_reasons[nf.get("reason", "unknown")] += 1
            for it in review_items:
                rt_categories[it.get("category", "UNKNOWN")] += 1
                rt_kinds[it.get("kind", "unknown")] += 1

        def _rate(num: int, den: int) -> float:
            return round(num / den, 4) if den else 0.0

        report: Dict[str, Any] = {
            "yield": {
                "attempted": attempted,
                "dropped_generation_errors": len(failures),
                "generated_ok": len(records),
                "flagged": len(flagged),
                "excluded": len(flagged) if auto_mode else 0,
                "kept": kept,
                "auto_mode": auto_mode,
                "rates": {
                    "generation_drop_rate": _rate(len(failures), attempted),
                    "qc_exclusion_rate": _rate(len(flagged), len(records)),
                    "overall_yield": _rate(kept, attempted),
                },
            },
            "generation_drops": {
                "total": len(failures),
                "by_stage": dict(drop_stage.most_common()),
                "by_error_type": dict(drop_type.most_common()),
                "by_stage_and_error": [
                    {"stage": s, "error_type": e, "count": c}
                    for (s, e), c in drop_stage_error.most_common()
                ],
            },
            "qc_exclusions": {
                "total": len(flagged),
                "by_mechanism": dict(mech.most_common()),
                "deterministic_vs_redteam_overlap": {
                    "deterministic_only": overlap["deterministic_only"],
                    "redteam_only": overlap["redteam_only"],
                    "both": overlap["both"],
                    "deterministic_total": overlap["deterministic_only"]
                    + overlap["both"],
                    "redteam_total": overlap["redteam_only"] + overlap["both"],
                },
            },
            "by_label": {
                "deterministic_leaks": dict(leak_labels.most_common()),
                "negative_integrity_by_label": dict(neg_labels.most_common()),
                "negative_integrity_by_reason": dict(neg_reasons.most_common()),
                "redteam_findings_by_category": dict(rt_categories.most_common()),
                "redteam_findings_by_kind": dict(rt_kinds.most_common()),
            },
        }
        path = self.write_json(report, filename)
        return path, report
