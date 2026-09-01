"""Tests that use only deterministic synthetic records."""

import unittest

import pandas as pd

import chemo_regimen_pipeline as pipeline
from generate_synthetic_data import build_synthetic_frame


class LabelTests(unittest.TestCase):
    def test_treatment_indicator_uses_only_explicit_yes_no(self):
        self.assertEqual(pipeline.normalize_treatment_indicator("Yes"), "Treatment")
        self.assertEqual(pipeline.normalize_treatment_indicator("none"), "No treatment")
        self.assertTrue(pd.isna(pipeline.normalize_treatment_indicator("not applicable")))
        self.assertTrue(pd.isna(pipeline.normalize_treatment_indicator("missing")))

    def test_non_treatment_is_not_a_regimen_family(self):
        for value in ["not applicable", "no", "none", "not reported", None]:
            with self.subTest(value=value):
                self.assertTrue(pd.isna(pipeline.classify_regimen(value)))

    def test_known_synthetic_regimen_examples(self):
        self.assertEqual(pipeline.classify_regimen("FOLFOX-6"), "FOLFOX/XELOX family")
        self.assertEqual(pipeline.classify_regimen("FOLFIRI"), "FOLFIRI family")
        self.assertEqual(
            pipeline.classify_regimen("Carboplatin + Paclitaxel"),
            "Carboplatin-based",
        )


class DatasetPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = build_synthetic_frame()
        cls.prepared = pipeline.prepare_feature_dataframe(cls.raw)

    def test_all_tasks_use_synthetic_rows_without_leakage_features(self):
        for spec in pipeline.TASKS:
            with self.subTest(task=spec.key):
                X, y, groups, feature_cols, audit = pipeline.create_task_dataset(
                    self.raw,
                    self.prepared,
                    spec,
                )
                self.assertGreater(len(y), 0)
                self.assertEqual(len(X), len(y))
                self.assertEqual(len(groups), len(y))
                self.assertGreaterEqual(groups.nunique(), 2)
                self.assertEqual(audit["usable_rows"], len(y))
                self.assertNotIn(pipeline.STUDY_COL, feature_cols)
                self.assertFalse(set(feature_cols) & pipeline.FORBIDDEN_FEATURE_COLS)
                self.assertNotIn(spec.indicator_col, feature_cols)
                self.assertNotIn(spec.regimen_col, feature_cols)

    def test_preprocessor_can_fit_synthetic_features(self):
        spec = pipeline.TASKS[0]
        X, y, _, feature_cols, _ = pipeline.create_task_dataset(
            self.raw,
            self.prepared,
            spec,
        )
        transformed = pipeline.build_preprocessor(feature_cols).fit_transform(X, y)
        self.assertEqual(transformed.shape[0], len(y))
        self.assertGreater(transformed.shape[1], 0)


if __name__ == "__main__":
    unittest.main()
