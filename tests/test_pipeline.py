"""Software regression tests for preprocessing and the two-stage pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data" / "preprocessing"))

from GenoExpSurv_main import (  # noqa: E402
    DEFAULTS,
    load_bundle,
    make_split,
    predict_arrays,
    read_manifest,
    run_split,
    seed_for_split,
)
from processing import prepare  # noqa: E402
from util import Cohort, FeatureBuilder, generate_demo  # noqa: E402


def test_two_stage_training_and_reload(tmp_path):
    dataset = generate_demo(tmp_path / "cohort", n=72, genes=18, seed=12)
    cohort = Cohort.load(dataset)
    split = make_split(cohort, seed=11)
    config = DEFAULTS | {
        "predictor_epochs": 4,
        "predictor_patience": 3,
        "mapper_epochs": 4,
        "mapper_patience": 3,
        "ridge_lambdas": [0.1],
    }
    metrics = run_split(cohort, split, config, tmp_path / "run")
    assert np.isfinite(metrics["c_index"])
    assert (tmp_path / "run" / "semantic_relevance.csv").exists()
    builder, anchor, model, baseline, mapper, _ = load_bundle(tmp_path / "run" / "model")
    arrays = builder.transform(cohort, split["test"])
    frame, relevance, weighted = predict_arrays(builder, anchor, model, baseline, mapper, arrays)
    assert len(frame) == len(split["test"])
    assert relevance.shape == weighted.shape
    assert (frame["survival_12m"].between(0, 1)).all()
    assert (frame["survival_24m"] <= frame["survival_12m"]).all()


def test_named_split_manifest(tmp_path):
    cohort = Cohort.load(generate_demo(tmp_path / "data", n=72, genes=12, seed=8))
    split = make_split(cohort, 11)
    rows = [dict(split_id="holdout_A", patient_id=pid, partition=part)
            for part, ids in split.items() for pid in ids]
    path = tmp_path / "splits.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    assert read_manifest(path, cohort, "holdout_A") == split
    assert seed_for_split(DEFAULTS, "holdout_A") == seed_for_split(DEFAULTS, "holdout_A")
    assert seed_for_split(DEFAULTS, "0") == DEFAULTS["seeds"][0]


def test_input_rejects_negative_tmb_and_empty_patient_id(tmp_path):
    folder = generate_demo(tmp_path / "data", n=30, genes=12, seed=5)
    patients = pd.read_csv(folder / "patients.csv")
    patients.loc[0, "TMB"] = -3
    patients.to_csv(folder / "patients.csv", index=False)
    with pytest.raises(ValueError, match="TMB must be nonnegative"):
        Cohort.load(folder)
    patients.loc[0, "TMB"] = 3
    patients.loc[0, "patient_id"] = ""
    patients.to_csv(folder / "patients.csv", index=False)
    with pytest.raises(ValueError, match="patient_id must be unique and nonempty"):
        Cohort.load(folder)


def test_source_preprocessing_with_audit(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    cp = pd.DataFrame([
        {"PATIENT_ID": f"P{i}", "OS_MONTHS": str(3 + i),
         "OS_STATUS": "1:DECEASED" if i % 2 else "0:LIVING",
         "SEX": "Male", "AGE_GROUP": "<65", "DRUG_TYPE": "anti-PD1"}
        for i in range(6)
    ])
    cs = pd.DataFrame([
        {"PATIENT_ID": f"P{i}", "SAMPLE_ID": f"S{i}", "CANCER_TYPE": "Melanoma",
         "SAMPLE_TYPE": "Primary", "GENE_PANEL": "IMPACT341",
         "TMB_NONSYNONYMOUS": "5.0"} for i in range(6)
    ])
    maf = pd.DataFrame([
        {"Tumor_Sample_Barcode": "S1", "Hugo_Symbol": "TP53",
         "Variant_Classification": "Missense_Mutation"},
        {"Tumor_Sample_Barcode": "S2", "Hugo_Symbol": "TP53",
         "Variant_Classification": "Silent"},
    ])
    sv = pd.DataFrame([{"Sample_Id": "S3", "Site1_Hugo_Symbol": "TP53",
                        "Site2_Hugo_Symbol": "EGFR"}])
    for name, df in (("patient.txt", cp), ("sample.txt", cs),
                     ("mutations.txt", maf), ("sv.txt", sv)):
        df.to_csv(source / name, sep="\t", index=False)
    pd.DataFrame([{"panel": "IMPACT341", "gene": "TP53"},
                  {"panel": "IMPACT341", "gene": "EGFR"}]).to_csv(
                      source / "panel.csv", index=False)
    with (source / "hallmark.gmt").open("w") as f:
        for i in range(50):
            f.write(f"HALLMARK_TEST_{i:02d}\ttest\tTP53\tEGFR\n")
    args = SimpleNamespace(
        out=str(tmp_path / "processed"), clinical_patient=str(source / "patient.txt"),
        clinical_sample=str(source / "sample.txt"), mutation_maf=str(source / "mutations.txt"),
        sv_calls=str(source / "sv.txt"), panel_genes=str(source / "panel.csv"),
        hallmark_gmt=str(source / "hallmark.gmt"), sample_policy="error", zero_time="error",
        panel_assignment=None, panel_source="error", ignore_unmapped=False,
        sv_status=None, write_local_pkl=False,
    )
    audit = prepare(args)
    cohort = Cohort.load(args.out)
    assert len(cohort.patients) == 6
    assert len(cohort.mutations) == 1
    assert len(cohort.sv) == 2
    assert json.loads((Path(args.out) / "conversion_audit.json").read_text())["hallmark_sets"] == 50
    fb = FeatureBuilder().fit(cohort, ["P0", "P1", "P2", "P3"])
    assert fb.transform(cohort, ["P4", "P5"]).genomic.shape[0] == 2
