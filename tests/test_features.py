import numpy as np
import pandas as pd
import pytest

from f1_podium.data import load_qualifying, load_results, load_standings
from f1_podium.features import FEATURES, build_features
from f1_podium.model import top3_hits


@pytest.fixture(scope="module")
def tables():
    results = load_results(first_year=2021)
    d, c = load_standings()
    return results, load_qualifying(), d, c


def test_no_missing_features(tables):
    df = build_features(*tables)
    assert not df[FEATURES].isna().any().any()


def test_features_do_not_use_the_race_outcome(tables):
    """Scrambling a race's result and standings must not change that race's features."""
    results, quali, d, c = tables
    race = results[results["year"] == 2022]["raceId"].iloc[0]
    base = build_features(results, quali, d, c)

    rng = np.random.default_rng(0)
    scrambled = results.copy()
    mask = scrambled["raceId"] == race
    scrambled.loc[mask, "positionOrder"] = rng.permutation(
        scrambled.loc[mask, "positionOrder"].values
    )
    scrambled["podium"] = (scrambled["positionOrder"] <= 3).astype(int)
    scrambled.loc[mask, "finished"] = ~scrambled.loc[mask, "finished"]
    d2 = d.copy()
    d2.loc[d2["raceId"] == race, "points"] += 100
    c2 = c.copy()
    c2.loc[c2["raceId"] == race, "points"] += 100
    after = build_features(scrambled, quali, d2, c2)

    key = ["raceId", "driverId"]
    a = base[base["raceId"] == race].sort_values(key)[FEATURES].reset_index(drop=True)
    b = after[after["raceId"] == race].sort_values(key)[FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_standings_come_from_the_previous_round(tables):
    df = build_features(*tables)
    first_rounds = df[df["round"] == 1]
    assert (first_rounds["driver_points_share"] == 0).all()
    assert (first_rounds["driver_champ_pos"] == 25).all()


def test_top3_hits():
    df = pd.DataFrame(
        {
            "raceId": [1] * 5,
            "podium": [1, 1, 1, 0, 0],
            "score": [0.9, 0.1, 0.8, 0.7, 0.2],
        }
    )
    assert top3_hits(df, "score").loc[1] == 2
