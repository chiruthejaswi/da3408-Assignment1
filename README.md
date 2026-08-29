# AIOps Module 1 Assignment 

**Author:** Chiruthejaswi Aramati

## Where to find each answer

### Q1 — Technical Debt Diagnosis (10 marks)
- Compact version: [`AIOps_Module1_Writeup.pdf`](./AIOps_Module1_Writeup.pdf) (Q1 section, page 1)
- Detailed version: [`Q1_Detailed_Diagnosis.pdf`](./Q1_Detailed_Diagnosis.pdf) 

### Q2 — MLflow Experiment Comparison (15 marks)
- Training script with MLflow logging: [`mlflow_train.py`](./mlflow_train.py)
- Run-comparison screenshot, written analysis, and logging code excerpt: [`AIOps_Module1_Writeup.pdf`](./AIOps_Module1_Writeup.pdf) (Q2 section) or [`Q1.pdf`](./Q1.pdf) / [`Q1.tex`](./Q1.tex)

### Q3 — DVC Data Versioning & Rollback (10 marks)
- DVC-tracked files: [`data.dvc`](./data.dvc), [`file_list.csv.dvc`](./file_list.csv.dvc)
- Remote config: [`.dvc/config`](./.dvc/config) (S3 remote)
- Version history: see git tags `v1` and `v2` — run `git log --oneline --tags` or `git tag -l`
- Rollback proof (terminal output): [`AIOps_Module1_Writeup.pdf`](./AIOps_Module1_Writeup.pdf) (Q3 section)

### Q4 — End-to-End Reproducibility Drill (15 marks)
- Completed collaboratively in a separate shared repository with partner (Ananya):
  **https://github.com/AnanyaKishore/q4demo**
- My role: Partner B — reproduced Partner A's Iris model run using `git clone`, `git checkout <commit>`, `dvc checkout`, `mamba env create -f environment.yml`, and reran `question_4.ipynb`. Verification note logged as an MLflow tag on the reproduced run documenting metric match within tolerance.

## Repo structure
```
.
├── README.md                        <- this file
├── AIOps_Module1_Writeup.pdf        <- Q1-Q4 combined 1-page write-up
├── Q1_Detailed_Diagnosis.pdf/.tex   <- detailed Q1 answer
├── Q1.pdf / Q1.tex                  <- LaTeX version of main write-up
├── mlflow_train.py                  <- Q2 training script with MLflow logging
├── data/                            <- Q3 dataset (DVC-tracked, not in git directly)
├── data.dvc, file_list.csv.dvc      <- DVC pointer files
├── file_list.csv                    <- CSV of filenames (DVC-tracked)
└── .dvc/                            <- DVC config (S3 remote)
```
