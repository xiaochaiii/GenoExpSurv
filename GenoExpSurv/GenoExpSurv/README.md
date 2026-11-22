# GenoExpSurv

**GenoExpSurv: Panel-Aware Multimodal Survival Prediction After Immune Checkpoint Inhibitors**

GenoExpSurv is a multimodal survival modeling framework for patients treated with immune checkpoint inhibitors (ICIs). It combines clinical and genomic information, sequencing-panel awareness, a Ridge-Cox-anchored survival predictor, and semantic interpretation of patient-level risk factors.

## Folder Specification

```text
GenoExpSurv/
├── README.md
├── requirements.txt
├── requirements-optional.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── default.yaml                # Default experiment settings
│   └── quick.yaml                  # Example configuration
├── data/
│   ├── raw/                        # Input data downloaded locally
│   ├── processed/                  # Generated analysis files
│   └── preprocessing/
│       ├── processing.py           # Clinical and genomic data processing
│       ├── panel_mask.py           # Sequencing-panel gene coverage
│       ├── get_Hallmark.py         # Hallmark gene-set validation
│       ├── matrix.py               # Partition-specific feature matrices
│       └── reference/              # Reference dictionaries and mappings
├── src/
│   ├── GenoExpSurv_main.py         # Training, prediction, and evaluation
│   ├── GenoExpSurv_models.py       # GenoExpSurv model components
│   ├── util.py                     # Features, survival metrics, and figures
│   └── seed.py                     # Random seed utilities
└── tests/
    └── test_pipeline.py           # Pipeline tests
```

## Model Framework

The framework comprises three components:

1. **Panel-aware multimodal representation:** integrates clinical variables, tumor mutational burden (TMB), SNV/Indel mutations, structural variants/fusions, sequencing-panel coverage, and 50 MSigDB Hallmark gene sets. The patient representation has 128 dimensions (32 clinical, 64 genomic, and 32 interaction).
2. **Survival prediction:** combines a Ridge-Cox risk anchor with a bounded neural residual. A training-derived Breslow baseline produces time-specific survival probability estimates.
3. **Semantic Mapper:** learns relevance weights over interpretable clinical and genomic aspects after the survival predictor has been fitted and frozen.

## Dataset Preparation

The primary dataset is the **MSKCC immunotherapy cohort** associated with Samstein *et al.* and distributed through [cBioPortal (`tmb_mskcc_2018`)](https://www.cbioportal.org/study/summary?id=tmb_mskcc_2018).

Download the study from cBioPortal and extract it to `data/raw/tmb_mskcc_2018/` with the following files:

```text
data/raw/tmb_mskcc_2018/
├── data_clinical_patient.txt
├── data_clinical_sample.txt
├── data_mutations.txt
├── data_sv.txt
└── data_gene_panel_matrix.txt
```

Additional inputs:

- `data/raw/panel_genes.csv`: sequencing-panel coverage table with columns `panel,gene` (obtain verified target gene lists through the panel utility or the original data source).
- `data/raw/hallmark.gmt`: the 50 Human Hallmark gene sets from [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp), subject to the provider's terms.

The `data/` directory contains shareable reference mapping assets and processing utilities. Study datasets are obtained from their original providers.

## Step 1: Environment Setup

Python 3.11 is recommended.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Check the command-line interface:

```bash
python src/GenoExpSurv_main.py --help
```

## Step 2: Data Processing

Create the sequencing-panel coverage reference:

```bash
python data/preprocessing/panel_mask.py --download-official --out-csv data/raw/panel_genes.csv
```

Download the MSigDB Hallmark GMT file to `data/raw/hallmark.gmt`, then validate it:

```bash
python data/preprocessing/get_Hallmark.py --gmt data/raw/hallmark.gmt
```

Process the MSKCC data:

```bash
python data/preprocessing/processing.py \
  --clinical-patient data/raw/tmb_mskcc_2018/data_clinical_patient.txt \
  --clinical-sample data/raw/tmb_mskcc_2018/data_clinical_sample.txt \
  --mutation-maf data/raw/tmb_mskcc_2018/data_mutations.txt \
  --sv-calls data/raw/tmb_mskcc_2018/data_sv.txt \
  --panel-genes data/raw/panel_genes.csv \
  --hallmark-gmt data/raw/hallmark.gmt \
  --zero-time keep \
  --out data/processed/mskcc_ici
```

This example preserves source-reported survival times. The preprocessing interface also provides `--panel-assignment`, `--panel-source`, and `--zero-time` options for explicitly selecting cohort-processing policies. Use settings consistent with the analysis protocol.

Validate the processed cohort:

```bash
python src/GenoExpSurv_main.py validate --data data/processed/mskcc_ici
```

## Step 3: Model Training

Train GenoExpSurv on one patient-level partition:

```bash
python src/GenoExpSurv_main.py train \
  --data data/processed/mskcc_ici \
  --config configs/default.yaml \
  --out outputs/experiment \
  --split 0
```

For ten independently generated patient-level partitions, use `--all-splits`. To use a supplied patient partition, provide `--manifest path/to/splits.csv` with `split_id`, `patient_id`, and `partition` columns.

## Step 4: Prediction and Evaluation

Generate survival predictions and semantic relevance outputs:

```bash
python src/GenoExpSurv_main.py predict \
  --data data/processed/mskcc_ici \
  --model outputs/experiment/split_0/model \
  --out outputs/predictions.csv
```

The pipeline supports survival ranking, time-dependent discrimination and probability-based metrics, risk stratification, calibration, subgroup analyses, and semantic visualization. The following commands are provided for methodological comparisons:

```bash
python src/GenoExpSurv_main.py benchmark --data data/processed/mskcc_ici --config configs/default.yaml --out outputs/baselines --split 0
python src/GenoExpSurv_main.py ablate --data data/processed/mskcc_ici --config configs/default.yaml --out outputs/ablation --split 0
```

External-cohort analyses using reduced-feature survival models are available through `python src/GenoExpSurv_main.py portable --help`.

## Code Check

A built-in demonstration and automated tests are available:

```bash
python src/GenoExpSurv_main.py self-test
python -m pytest -q tests/
```

## References

- Samstein, R. M., et al. *Tumor mutational load predicts survival after immunotherapy across multiple cancer types.* Nature Genetics (2019).
- [cBioPortal: TMB and Immunotherapy (MSK, 2018)](https://www.cbioportal.org/study/summary?id=tmb_mskcc_2018)
- [MSigDB Human Hallmark gene sets](https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp)
