# GenoExpSurv

**Panel-Aware Multimodal Survival Prediction After Immune Checkpoint Inhibitors**

GenoExpSurv integrates clinical and genomic features for overall survival prediction after immune checkpoint inhibitor therapy. It combines panel-aware encoding, a Ridge-Cox-anchored residual predictor, and a Semantic Mapper.

## Project Structure

```text
GenoExpSurv/
├── configs/              # Experiment settings
├── data/preprocessing/   # Data preparation
├── src/                  # Model, training and evaluation
├── tests/                # Code tests
├── requirements.txt
└── README.md
```

## Dataset Preparation

Download the [MSKCC immunotherapy cohort (`tmb_mskcc_2018`)](https://www.cbioportal.org/study/summary?id=tmb_mskcc_2018) and place its files in `data/raw/tmb_mskcc_2018/`:

- `data_clinical_patient.txt`
- `data_clinical_sample.txt`
- `data_mutations.txt`
- `data_sv.txt`
- `data_gene_panel_matrix.txt`

Prepare `data/raw/panel_genes.csv` (panel gene coverage) and download `data/raw/hallmark.gmt` (50 Human Hallmark gene sets) from [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp).

## Step 1: Process Data

Install the requirements (Python 3.11 recommended):

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Generate the panel gene reference if needed:

```bash
python data/preprocessing/panel_mask.py --download-official --out-csv data/raw/panel_genes.csv
```

Process the cohort:

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

## Step 2: Run GenoExpSurv

Train:

```bash
python src/GenoExpSurv_main.py train --data data/processed/mskcc_ici --config configs/default.yaml --out outputs/experiment --split 0
```

Predict:

```bash
python src/GenoExpSurv_main.py predict --data data/processed/mskcc_ici --model outputs/experiment/split_0/model --out outputs/predictions.csv
```

For other commands, use `python src/GenoExpSurv_main.py --help`.

## Reference

Samstein, R. M., et al. *Tumor mutational load predicts survival after immunotherapy across multiple cancer types.* **Nature Genetics** (2019).
