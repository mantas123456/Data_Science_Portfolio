# 📄 CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [1.2.0] - 2025-04-09
### 🚀 Added
- Four new features for model input:
  - `laps_completed`
  - `fastest_lap_rank`
  - `constructor_standing_position`
  - `year_bin` (season grouping)
- Feature importance plots (Random Forest & XGBoost)
- Model performance bar chart visualization
- Enhanced model evaluation using reusable utility functions
- Confusion matrix grid comparison
- Updated `README.md` with results, visuals, and versioning

### ✅ Improved
- Model recall and F1-score with richer feature set
- Notebook modularity and commenting
- Environment reproducibility via `environment.yml`

---

## [1.1.0] - 2025-04-08
### 🚀 Added
- Confusion matrix plotting utility in `utils/evaluate_models.py`
- Evaluation table comparison across multiple models
- Git tagging and versioning structure
- First professional README draft with metric summary

---

## [1.0.0] - 2025-04-07
### 🎉 Initial Release
- Set up project structure and GitHub repo
- Loaded and cleaned Formula 1 dataset
- Built baseline models (Random Forest, Logistic Regression, XGBoost)
- Created first model evaluation and confusion matrix
- Implemented Conda environment and `.gitignore`
