from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from refined_c4_recurrent_class_analysis import (
    DEFAULT_FAMILY_ARTIFACTS,
    ROOT,
    analyze_family,
    family_artifact_map,
)


DEFAULT_CLASS_ARTIFACT = ROOT / "artifacts" / "refined_c4_nonrecurrent_survivors.json"


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def class_profile(record: Dict[str, object]) -> Tuple[int, ...]:
    return tuple(int(value) for value in record["col_sum_profile"].split(","))


def class_sort_key(record: Dict[str, object]) -> Tuple[int, str]:
    label = record["label"]
    if label.startswith("U") and label[1:].isdigit():
        return (int(label[1:]), label)
    return (10**9, label)


def load_selected_records(path: Path, labels: Sequence[str] | None) -> List[Dict[str, object]]:
    data = load_json(path)
    records = sorted(data["classes"], key=class_sort_key)
    if not labels:
        return records

    wanted = set(labels)
    selected = [record for record in records if record["label"] in wanted]
    missing = sorted(wanted - {record["label"] for record in selected})
    if missing:
        raise ValueError(f"nonrecurrent class label(s) not found in {path}: {missing}")
    return selected


def summarize_global(analyses: Sequence[Dict[str, object]]) -> Dict[str, object]:
    totals: Dict[str, object] = {
        "class_count": len(analyses),
        "canonical_matrix_count": 0,
        "bipartite_count": 0,
        "three_connected_count": 0,
        "abstract_planar_count": 0,
        "abstract_nonplanar_count": 0,
        "kuratowski_core_counts": {},
        "family_support": {},
        "profile_support": {},
    }
    for analysis in analyses:
        totals["canonical_matrix_count"] += analysis["canonical_matrix_count"]
        totals["bipartite_count"] += analysis["bipartite_count"]
        totals["three_connected_count"] += analysis["three_connected_count"]
        totals["abstract_planar_count"] += analysis["abstract_planar_count"]
        totals["abstract_nonplanar_count"] += analysis["abstract_nonplanar_count"]
        family = analysis["family"]
        profile = ",".join(str(value) for value in analysis["col_sums"])
        totals["family_support"][family] = totals["family_support"].get(family, 0) + 1
        totals["profile_support"][profile] = totals["profile_support"].get(profile, 0) + 1
        for key, value in analysis["kuratowski_core_counts"].items():
            totals["kuratowski_core_counts"][key] = totals["kuratowski_core_counts"].get(key, 0) + value
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze one or more nonrecurrent refined-C4 obstruction classes across their supported families."
    )
    parser.add_argument(
        "--class-artifact",
        type=Path,
        default=DEFAULT_CLASS_ARTIFACT,
        help="Path to artifacts/refined_c4_nonrecurrent_survivors.json.",
    )
    parser.add_argument(
        "--label",
        action="append",
        help="Nonrecurrent class label to analyze (may be repeated). Defaults to all U-classes.",
    )
    parser.add_argument(
        "--family-artifact",
        action="append",
        dest="family_artifacts",
        help="Path to a refined-C4 family obstruction artifact (may be repeated).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "refined_c4_nonrecurrent_class_analysis.json",
        help="Path to write the class-level JSON artifact.",
    )
    args = parser.parse_args()

    records = load_selected_records(args.class_artifact, args.label)
    family_paths = tuple(Path(path) for path in (args.family_artifacts or DEFAULT_FAMILY_ARTIFACTS))
    family_map = family_artifact_map(family_paths)

    class_payloads: List[Dict[str, object]] = []
    analyses: List[Dict[str, object]] = []
    for record in records:
        family_name = record["family_name"]
        if family_name not in family_map:
            raise ValueError(f"no family artifact provided for {family_name}")
        analysis = analyze_family(
            family_map[family_name],
            record["label"],
            class_profile(record),
            tuple(tuple(edge) for edge in record["representative_internal_edges"]),
        )
        class_payloads.append(
            {
                "class_label": record["label"],
                "nonrecurrent_class_record": record,
                "analysis": analysis,
            }
        )
        analyses.append(analysis)

    out_path = args.out.resolve()
    payload = {
        "source_class_artifact": str(args.class_artifact.relative_to(ROOT)) if args.class_artifact.is_absolute() else str(args.class_artifact),
        "source_family_artifacts": [str(path.relative_to(ROOT)) if path.is_absolute() else str(path) for path in family_paths],
        "selected_labels": [record["label"] for record in records],
        "classes": class_payloads,
        "global": summarize_global(analyses),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
