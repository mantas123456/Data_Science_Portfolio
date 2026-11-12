"""Models and race-level evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .features import FEATURES


class GridBaseline:
    """Predicts that the front three on the grid finish on the podium."""

    name = "Baseline: top 3 on grid"
    probabilistic = False

    def fit(self, X, y):
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return -X["grid"].to_numpy()


class LogitModel:
    name = "Logistic regression"
    probabilistic = True

    def __init__(self, seed: int = 42):
        self.m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

    def fit(self, X, y):
        self.m.fit(X[FEATURES], y)
        return self

    def score(self, X):
        return self.m.predict_proba(X[FEATURES])[:, 1]


class XGBModel:
    name = "XGBoost"
    probabilistic = True

    def __init__(self, seed: int = 42):
        self.m = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
        )

    def fit(self, X, y):
        self.m.fit(X[FEATURES], y)
        return self

    def score(self, X):
        return self.m.predict_proba(X[FEATURES])[:, 1]


def top3_hits(df: pd.DataFrame, score_col: str) -> pd.Series:
    """For each race, how many of the 3 highest-scored drivers actually made the podium."""
    ranked = df.sort_values(["raceId", score_col], ascending=[True, False])
    top = ranked.groupby("raceId").head(3)
    return top.groupby("raceId")["podium"].sum()


def evaluate(test: pd.DataFrame, models: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return overall metrics and hits-per-race by season for each model."""
    rows, seasons = [], []
    for m in models:
        test = test.assign(score=m.score(test))
        hits = top3_hits(test, "score")
        row = {
            "model": m.name,
            "podium_picks_correct": round(hits.mean() / 3, 3),
            "all_three_correct": round(float((hits == 3).mean()), 3),
            "roc_auc": round(roc_auc_score(test["podium"], test["score"]), 3),
        }
        if m.probabilistic:
            row["log_loss"] = round(log_loss(test["podium"], test["score"]), 3)
            row["brier"] = round(brier_score_loss(test["podium"], test["score"]), 3)
        rows.append(row)
        by_year = hits.to_frame("hits").join(test.groupby("raceId")["year"].first())
        s = by_year.groupby("year")["hits"].mean().div(3).round(3).rename(m.name)
        seasons.append(s)
    return pd.DataFrame(rows), pd.concat(seasons, axis=1)
