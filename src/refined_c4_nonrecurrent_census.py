from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from refined_c4_recurrent_skeletons import graph_invariants, load_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts" / "refined_c4_k33_templates.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "refined_c4_nonrecurrent_survivors.json"


def summarize_nonrecurrent_classes(data: Dict[str, object]) -> Dict[str, object]:
    classes = [
        skeleton
        for skeleton in data["global"]["internal_skeleton_classes"]
        if skeleton["count"] == 1
    ]

    labeled: List[Dict[str, object]] = []
    family_support: Dict[str, int] = {}
    profile_support: Dict[str, int] = {}
    family_profile_support: Dict[str, int] = {}

    for index, skeleton in enumerate(classes, start=1):
        family_names = sorted(skeleton["families"])
        profiles = sorted(skeleton["col_sum_profiles"])
        if len(family_names) != 1 or len(profiles) != 1:
            raise ValueError(
                f"expected one-off class to have one family/profile, got families={family_names}, profiles={profiles}"
            )
        family_name = family_names[0]
        profile = profiles[0]
        family_support[family_name] = family_support.get(family_name, 0) + 1
        profile_support[profile] = profile_support.get(profile, 0) + 1
        family_profile_key = f"{family_name} | {profile}"
        family_profile_support[family_profile_key] = family_profile_support.get(family_profile_key, 0) + 1

        labeled.append(
            {
                "label": f"U{index}",
                "source_class_id": skeleton["class_id"],
                "family_name": family_name,
                "col_sum_profile": profile,
                "representative_internal_edges": skeleton["representative_internal_edges"],
                "invariants": graph_invariants(skeleton["representative_internal_edges"]),
                "example": skeleton["examples"][0],
            }
        )

    return {
        "source_artifact": str(DEFAULT_INPUT.relative_to(ROOT)),
        "nonrecurrent_class_count": len(labeled),
        "nonrecurrent_survivor_count": len(labeled),
        "family_support": family_support,
        "profile_support": profile_support,
        "family_profile_support": family_profile_support,
        "classes": labeled,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the nonrecurrent refined-C4 8-vertex survivors from the K3,3 template artifact."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to artifacts/refined_c4_k33_templates.json.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the nonrecurrent survivor census.",
    )
    args = parser.parse_args()

    payload = summarize_nonrecurrent_classes(load_json(args.input))
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
