# 🏎️ Formula 1 Podium Prediction

This project uses historical Formula 1 race data to build machine learning models that predict whether a driver will finish on the podium (Top 3). The pipeline includes feature engineering, model training, comparison, and interpretability analysis.

---

## 📁 Project Structure


---

## 🎯 Objective

To build a predictive model that estimates whether a Formula 1 driver will finish in the **top 3** based on:
- Starting grid position
- Qualifying performance
- Driver and constructor historical performance
- Year and team data

---

## 📊 Data Sources

The dataset includes detailed records for:
- Races, drivers, constructors
- Qualifying sessions
- Results and standings
- Pit stops and lap times (not used in baseline model)

---

## 🧠 Feature Engineering

The following features were engineered and selected for modeling:
- `grid`: Starting position on the grid
- `qualifying_position`: Best qualifying rank
- `driver_points`: Driver’s season points before the race
- `constructor_points`: Constructor’s season points before the race
- `driverRef` and `constructor_name`: Categorical identifiers
- `year`: Season of the race

---

## 🤖 Models Used

- **Random Forest Classifier**
- **Logistic Regression**
- **XGBoost Classifier**

---

## 🏆 Results Summary

| Model             | Precision | Recall | F1 Score |
|------------------|-----------|--------|----------|
| Random Forest     | ~0.65     | ~0.50  | ~0.56    |
| Logistic Regression | ~0.68   | ~0.52  | ~0.58    |
| **XGBoost**       | **~0.67** | **~0.54** | **~0.60** |

**✅ XGBoost showed the best overall performance.**

---

## 🧠 Feature Importance Analysis

### Top Predictive Features:
- `grid`, `qualifying_position`, `driver_points`, `constructor_points`  
→ All highly correlated with podium finishes, which aligns with real-world F1 dynamics.

### Model Insights:
- Random Forest emphasized **constructor_points**
- XGBoost captured more nuanced **driver-team-year** interactions
- Categorical features (like driver/team) were more useful for XGBoost

---

## 📈 Visualizations

- Bar charts comparing **precision/recall/F1** for all models
- Feature importance plots (Random Forest vs XGBoost)

All plots are saved in the `results/` directory.

---

## 🧰 Tools & Libraries

- `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`
- `xgboost`
- Python 3.11 (via Anaconda)

---

## 🔍 Confusion Matrix Comparison

This visualization compares the classification performance of all three models:

- ✅ **True Positives (Bottom-right)**: Correctly predicted podium finishes
- ⚠️ **False Negatives (Bottom-left)**: Real podiums missed by the model
- ✅ **True Negatives (Top-left)**: Correctly identified non-podiums

![Confusion Matrix Comparison](results/confusion_matrix_comparison.png)

---

## 📌 Version Control

This project uses **Git + GitHub** for version control. To start tracking:

```bash
git init
git add .
git commit -m "Initial commit - F1 Podium Predictor"
git remote add origin https://github.com/your-username/Data_Science_Portfolio.git
git push -u origin main

---