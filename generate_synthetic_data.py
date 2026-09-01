"""Generate a deterministic fictional workbook for the public demo.

The rows are created from distributions and fixed vocabularies in this file.
They are not sampled, perturbed, or otherwise derived from a private dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from chemo_regimen_pipeline import (
    ADJ_INDICATOR_COL,
    ADJ_REGIMEN_COL,
    NEO_INDICATOR_COL,
    NEO_REGIMEN_COL,
    STUDY_COL,
)


def build_synthetic_frame(n_rows: int = 120, seed: int = 42) -> pd.DataFrame:
    """Return fictional records with enough class support for the demo."""
    if n_rows < 80:
        raise ValueError("n_rows must be at least 80 for stable demonstration folds.")

    rng = np.random.default_rng(seed)
    row = np.arange(n_rows)
    studies = np.array(["synthetic-cohort-a", "synthetic-cohort-b", "synthetic-cohort-c", "synthetic-cohort-d"])
    diagnoses = np.array(["synthetic colorectal", "synthetic breast", "synthetic gastric"])
    regimen_examples = np.array(
        [
            "FOLFOX-6",
            "FOLFIRI",
            "Carboplatin + Paclitaxel",
            "EC + Docetaxel",
        ]
    )

    neo_treated = row % 3 != 0
    adj_treated = (row + 1) % 3 != 0
    neo_regimens = np.where(
        neo_treated,
        regimen_examples[row % len(regimen_examples)],
        "not applicable",
    )
    adj_regimens = np.where(
        adj_treated,
        regimen_examples[(row + 1) % len(regimen_examples)],
        "not applicable",
    )

    frame = pd.DataFrame(
        {
            "record_id": [f"SYN-{value:04d}" for value in row],
            STUDY_COL: studies[row % len(studies)],
            NEO_INDICATOR_COL: np.where(neo_treated, "yes", "no"),
            NEO_REGIMEN_COL: neo_regimens,
            ADJ_INDICATOR_COL: np.where(adj_treated, "yes", "no"),
            ADJ_REGIMEN_COL: adj_regimens,
            "sex": np.where(row % 2 == 0, "female", "male"),
            "age_at_diagnosis": rng.integers(35, 82, size=n_rows),
            "bmi": np.round(rng.normal(25, 3.5, size=n_rows), 1),
            "primary_diagnosis": diagnoses[row % len(diagnoses)],
            "sample_type": np.where(row % 2 == 0, "primary", "metastasis"),
            "sample_site": np.where(row % 3 == 0, "site-a", "site-b"),
            "histology": np.where(row % 4 == 0, "type-a", "type-b"),
            "tumor_side": np.where(row % 2 == 0, "left", "right"),
            "tumor_marker_1": np.where(
                row % 3 == 0,
                "pathological",
                "within normal range",
            ),
            "tumor_marker_2": np.where(
                row % 4 == 0,
                "pathological",
                "physiological",
            ),
            "comorbidities": np.where(
                row % 5 == 0,
                "synthetic condition",
                "none",
            ),
            "t_stage": np.array(["t1", "t2", "t3", "t4"])[row % 4],
            "n_stage": np.array(["n0", "n1", "n2"])[row % 3],
            "m_stage": np.where(row % 4 == 0, "m1", "m0"),
            "venous_invasion": np.where(row % 3 == 0, "v1", "v0"),
            "lymphatic_invasion": np.where(row % 4 == 0, "l1", "l0"),
            "perineural_invasion": np.where(row % 5 == 0, "pn1", "pn0"),
            "grade": np.array(["g1", "g2", "g3"])[row % 3],
            "resection_margin": np.where(row % 7 == 0, "r1", "r0"),
            "tumor_or_metastasis_diameter": np.round(
                rng.uniform(5, 85, size=n_rows),
                1,
            ),
            "metastasis_timing": np.where(
                row % 2 == 0,
                "synchronous",
                "metachronous",
            ),
            "intrahepatic_metastasis_count": rng.integers(0, 5, size=n_rows),
            "extrahepatic_metastasis_count": rng.integers(0, 3, size=n_rows),
            "extrahepatic_metastasis_site": np.where(
                row % 4 == 0,
                "synthetic-site",
                "none",
            ),
        }
    )
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("synthetic_demo.xlsx"))
    parser.add_argument("--rows", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    build_synthetic_frame(args.rows, args.seed).to_excel(destination, index=False)
    print(f"Synthetic demo workbook written to: {destination}")


if __name__ == "__main__":
    main()
