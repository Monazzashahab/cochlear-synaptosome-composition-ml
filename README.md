# Cochlear synaptosome proteomics: nested machine learning and composition analysis

This repository contains the complete analysis code for the article:

> Shahab M. Preparation composition confounds treatment effects in cochlear synaptosomal proteomics: a nested machine learning reanalysis of cisplatin and MnTBAP exposure. Research Square. Preprint, 2026. [doi:10.21203/rs.3.rs-11340135/v1](https://doi.org/10.21203/rs.3.rs-11340135/v1)

The code reanalyzes a TMTpro 16-plex proteomics dataset of mouse cochlear synaptosomes from four groups (Control, Cisplatin, MnTBAP+Cisplatin and MnTBAP, four biological replicates each). It starts from the raw reporter ion abundances in the Proteome Discoverer export. It regenerates every number, table and figure in the article.

## What the pipeline does

1. **Channel quality control.** Excludes any channel whose total reporter abundance is below 25% of the median channel.
2. **Filtering and normalization.** Keeps High-confidence proteins that are quantified in at least three retained replicates of every group. Applies log2 transformation and per-channel median centering. No imputation.
3. **Differential abundance.** Uses empirical Bayes moderated t-statistics (a Python implementation of limma eBayes) with Benjamini-Hochberg FDR. The implementation is validated against InMoose.
4. **Global signal and permutation null.** Computes Storey π0 and refits the model on all 35 partitions of the Control and Cisplatin samples.
5. **Classification.**
   - Nested leave-one-out cross-validation, with the top 50 proteins re-selected inside every fold.
   - Five classifiers.
   - An exact permutation test over all 70 labelings.
   - A comparison with the biased protocol, where proteins are selected before cross-validation.
6. **Consensus biomarker ranking.** Combines selection stability, logistic regression, random forest and Hedges' g.
7. **Held-out projection of the MnTBAP groups** onto the Control to Cisplatin axis, with calibrated leave-one-out references.
8. **Tissue composition.**
   - An unsupervised principal component and six marker panels (synaptic, bone, blood, marrow, myelin and nuclear).
   - Differential abundance adjusted for composition, with alternative covariates.
9. **Four-group classification and co-expression modules.**
10. **Sensitivity analyses.** Sample influence, all partitions, a reduced design, and the effect of restoring the excluded channel.

## Repository contents

| File | Purpose |
|---|---|
| `p1_analysis.py` | Main pipeline (steps 1 to 9). Writes tables and `output/results.pkl`. |
| `p1_sensitivity.py` | Sensitivity analyses (step 10). Run after `p1_analysis.py`. |
| `p1_figures.py` | Figures 1 to 6 and S1 to S5 (PNG at 300 dpi and PDF). |
| `limma_stats.py` | Empirical Bayes moderated t-statistics and Benjamini-Hochberg FDR. |
| `validate_limma.py` | Checks `limma_stats.py` against InMoose (Python port of limma). |
| `relabel_outputs.py` | Renames channel codes (A1 to D4) in the output tables to the sample names used in the article (Control 1 to MnTBAP 4). Imported by the other scripts. |
| `requirements.txt` | Package versions used for the article. |
| `data/README.md` | Description of the input data and how to obtain them. |

## Installation

Python 3.11 was used.

```bash
git clone https://github.com/Monazzashahab/cochlear-synaptosome-composition-ml.git
cd cochlear-synaptosome-composition-ml
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

The input data are not included in this repository. See `data/README.md` for a description of the data, the original publications, and where to send access requests. Place the Proteome Discoverer export at `data/2363_MC-Loading_2.xlsx`, or set the environment variable `P1_DATA_FILE` to its path.

## How to run

```bash
python3 p1_analysis.py       # main pipeline, about 8 minutes on 2 CPU cores
python3 p1_sensitivity.py    # sensitivity analyses, about 2 minutes
python3 validate_limma.py    # validation against InMoose
python3 p1_figures.py        # figures
```

The classification and permutation results are cached in `output/bin_cache.pkl` after the first run. To recompute them, set `P1_RECOMPUTE=1`:

```bash
P1_RECOMPUTE=1 python3 p1_analysis.py
```

## Outputs

All outputs are written to `output/` (tables) and `figures/` (figures).

| Output | Content |
|---|---|
| `Table1_binary_classification.csv` | Nested and biased classification performance with permutation p values |
| `Table2_top20_consensus.csv` | Top 20 consensus-ranked proteins of the cisplatin signature |
| `Table3_heldout_classifier_calls.csv` | Classifier calls for the held-out MnTBAP groups |
| `Table4_module_summary.csv` | Co-expression module summary |
| `Table5_composition_panels.csv` | Composition panel scores by group |
| `Table6_adjusted_DE_summary.csv` | Differential abundance with and without adjustment for composition |
| `TableS1` to `TableS16` | Supplementary tables (channel QC, all proteins, signature, rankings, axis scores, modules, sensitivity, composition, covariate sensitivity) |
| `analysis_summary.json`, `sensitivity_summary.json` | Key numbers reported in the article |
| `limma_validation.txt` | Agreement with InMoose |
| `figures/Figure1` to `Figure6`, `FigureS1` to `FigureS5` | Article figures |

## Pre-specified analysis decisions

- Input: the `Abundances (Grouped)` reporter columns 126 to 134N only. Normalized values, group statistics, fold changes and p values computed in the original processing are not used.
- Channel quality control: exclusion below 25% of the median channel total.
- Proteins: Protein FDR confidence "High", quantified in at least three retained replicates of every group.
- Differential abundance: FDR below 0.05. Exploratory cisplatin signature: Cisplatin versus Control FDR below 0.20.
- Classification: top 50 proteins selected inside each fold. Classifier hyperparameters are fixed in advance.

## Reproducibility

- All random processes use a fixed seed (42), and PCA uses the full SVD solver, so repeated runs give identical results.
- The moderated statistics agree with InMoose to within 3 × 10⁻¹⁴ for the group model and 1.5 × 10⁻¹³ for the composition covariate model.

## Citation

If you use this code, please cite the article above. Citation details are also in `CITATION.cff`.

## License

The code is released under the MIT License (see `LICENSE`). The license covers the code only, not the data.

## Contact

Monazza Shahab, PhD, Independent Researcher, Columbus, OH, USA. monazzashahab@gmail.com
