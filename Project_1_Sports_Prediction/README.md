# 🏎️ Formula 1 Podium Prediction

This project uses historical Formula 1 race data to build machine learning models that predict whether a driver will finish on the **podium (Top 3)**. It demonstrates the full ML lifecycle: from data exploration and feature engineering to model training, evaluation, and visual interpretation.

---

## 📁 Project Structure

**Project_1_Sports_Prediction/**
- **data/**
  - Formula_1/
    - races.csv
    - results.csv
    - drivers.csv
    - qualifying.csv
    - driver_standings.csv
    - constructor_standings.csv
- **notebooks/**
  - 01_data_exploration.ipynb → Exploratory data analysis
  - 02_podium_prediction_model.ipynb → Model training, evaluation, visuals
- **utils/**
  - evaluate_models.py → Custom evaluation metrics + confusion matrix plotting
- **results/**
  - confusion_matrix_comparison.png → Confusion matrix for all models
- environment.yml → Conda environment setup (Python 3.11, scikit-learn, xgboost, etc.)
- .gitignore → Files/folders to exclude from Git
- README.md → Project overview, results, versioning info

---

## 🎯 Objective

To build a predictive model that estimates whether a Formula 1 driver will finish in the **top 3**, using features such as:
- Starting grid position
- Qualifying rank
- Driver and constructor season points
- Team and year data

---

## 📊 Data Sources

The dataset includes detailed records for:
- Races, drivers, constructors
- Qualifying sessions
- Final race results and season standings
- Pit stops and lap times (not used in baseline model)

---

## 🧠 Feature Engineering

Features selected for modeling:

| Feature              | Description                                      |
|----------------------|--------------------------------------------------|
| `grid`               | Starting position on the grid                    |
| `qualifying_position`| Best qualifying position from qualifying session |
| `driver_points`      | Driver’s total points before the race            |
| `constructor_points` | Constructor’s points before the race             |
| `driverRef`          | Driver identity (one-hot encoded)                |
| `constructor_name`   | Team name (one-hot encoded)                      |
| `year`               | Season year (one-hot encoded)                    |

---

## 🤖 Models Used

- Random Forest Classifier
- Logistic Regression (with feature scaling)
- XGBoost Classifier

---

## 🏆 Results Summary

| Model                | Precision | Recall | F1 Score |
|---------------------|-----------|--------|----------|
| Random Forest        | ~0.65     | ~0.50  | ~0.56    |
| Logistic Regression  | ~0.68     | ~0.52  | ~0.58    |
| **XGBoost**          | **~0.67** | **~0.54** | **~0.60** |

**✅ XGBoost provided the best overall balance between precision and recall.**

---

## 🔍 Feature Importance Analysis

Key takeaways:

- `grid`, `qualifying_position`, and `points` are strong predictors — aligning with F1 intuition.
- Random Forest emphasized `constructor_points`.
- XGBoost captured complex team/driver interactions.
- Categorical features (driver/team/year) added value, especially in XGBoost.

---

## 📈 Visualizations

- 📊 Bar chart comparing **precision**, **recall**, and **F1 score**
- 🔥 Feature importance plots (Random Forest vs XGBoost)

All plots are saved in the [`results/`](./results/) folder.

---

## 🔬 Confusion Matrix Comparison

Visual comparison of model predictions on podium classification:

- ✅ **True Positives (Bottom-right)**: Correct podium predictions
- ⚠️ **False Negatives (Bottom-left)**: Missed podiums
- ✅ **True Negatives (Top-left)**: Correct non-podium predictions

![Confusion Matrix Comparison](results/confusion_matrix_comparison.png)

---

## 🧰 Tools & Libraries

- `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`, `xgboost`
- Python 3.11 (Anaconda environment)

---

## 🧪 Evaluation Utilities

A reusable evaluation script is available in [`utils/evaluate_models.py`](./utils/evaluate_models.py), including:

- Precision, recall, F1, and accuracy comparison
- Confusion matrix heatmap visualizer

---

## 🧾 Version History

| Version | Date       | Summary                                                |
|---------|------------|--------------------------------------------------------|
| 1.0.0   | 2025-04-07 | Initial version: baseline models and feature analysis |
| 1.1.0   | 2025-04-08 | Added evaluation script, scaled logistic regression, confusion matrix visualization, and README update |

---

## 📌 Version Control

This project uses Git & GitHub for version control.

```bash
git init
git add .
git commit -m "Initial commit - F1 Podium Predictor"
git remote add origin https://github.com/your-username/Data_Science_Portfolio.git
git push -u origin main
