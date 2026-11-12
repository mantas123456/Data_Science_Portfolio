"""Train on earlier seasons, test on later ones, write results.

Usage: python -m f1_podium [--test-from 2023] [--out results]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .data import load_qualifying, load_results, load_standings  # noqa: E402
from .features import FEATURES, build_features  # noqa: E402
from .model import GridBaseline, LogitModel, XGBModel  # noqa: E402
from .model import evaluate as run_eval  # noqa: E402


def plot_importance(model: XGBModel, path: Path) -> None:
    imp = pd.Series(model.m.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp.index, imp.values, color="#2c6e8f")
    ax.set_xlabel("XGBoost feature importance (normalised)")
    ax.set_title("What predicts a podium, using pre-race information only", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_seasons(by_season: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    (by_season * 100).plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("Podium picks correct (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 100)
    ax.set_title(
        "Held-out seasons: share of the 3 picks per race that made the podium",
        loc="left",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="F1 podium prediction backtest")
    ap.add_argument("--test-from", type=int, default=2023, help="first held-out season")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d_st, c_st = load_standings()
    df = build_features(load_results(), load_qualifying(), d_st, c_st)
    train = df[df["year"] < args.test_from]
    test = df[df["year"] >= args.test_from]

    models = [GridBaseline(), LogitModel(), XGBModel()]
    for m in models:
        m.fit(train, train["podium"])
    overall, by_season = run_eval(test, models)

    overall.to_csv(out / "metrics.csv", index=False)
    by_season.to_csv(out / "podium_picks_by_season.csv")
    plot_importance(models[-1], out / "feature_importance.png")
    plot_seasons(by_season, out / "podium_picks_by_season.png")

    n_races = test["raceId"].nunique()
    print(f"Train: {train['raceId'].nunique()} races ({train['year'].min()}-{train['year'].max()})")
    print(f"Test:  {n_races} races ({test['year'].min()}-{test['year'].max()})\n")
    print(overall.to_string(index=False))
    print("\n" + by_season.to_string())


if __name__ == "__main__":
    main()
