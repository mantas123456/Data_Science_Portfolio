"""Pre-race features: only information available before the lights go out.

The v1 model used three things that are not known before a race: championship
standings *after* the race, laps completed, and fastest-lap rank. This module
builds every feature from qualifying (known on Saturday) and from earlier races.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "grid",
    "quali_pos",
    "driver_champ_pos",
    "driver_points_share",
    "team_champ_pos",
    "team_points_share",
    "driver_avg_finish_3",
    "driver_podium_rate_5",
    "driver_finish_rate_10",
    "team_best_finish_3",
    "teammate_quali_gap",
]

NO_INFO_POS = 25.0  # used for "no previous standing" and pit-lane starts


def _previous_standings(
    results: pd.DataFrame, standings: pd.DataFrame, key: str, prefix: str
) -> pd.DataFrame:
    """Attach standings from the previous round of the same season.

    Ergast standings for race r already include race r's points, so they are
    shifted to the next race. The first race of a season has no prior standing.
    """
    races = results[["raceId", "year", "round"]].drop_duplicates().sort_values(["year", "round"])
    races["next_raceId"] = races.groupby("year")["raceId"].shift(-1)
    s = standings.merge(races[["raceId", "next_raceId"]], on="raceId").dropna(
        subset=["next_raceId"]
    )
    s["raceId"] = s["next_raceId"].astype(int)
    leader = s.groupby("raceId")["points"].transform("max").replace(0, np.nan)
    s[f"{prefix}_points_share"] = (s["points"] / leader).fillna(0.0)
    s[f"{prefix}_champ_pos"] = s["position"].astype(float)
    return s[["raceId", key, f"{prefix}_points_share", f"{prefix}_champ_pos"]]


def _rolling_past(df: pd.DataFrame, by: str, col: str, window: int, how: str) -> pd.Series:
    """Rolling statistic over the previous `window` races, excluding the current one."""
    g = df.groupby(by)[col]
    shifted = g.shift(1)
    roll = shifted.groupby(df[by]).rolling(window, min_periods=1)
    out = roll.mean() if how == "mean" else roll.min()
    return out.reset_index(level=0, drop=True)


def build_features(
    results: pd.DataFrame, qualifying: pd.DataFrame, driver_st: pd.DataFrame, team_st: pd.DataFrame
) -> pd.DataFrame:
    df = results.sort_values(["date", "raceId", "positionOrder"]).reset_index(drop=True)
    df = df.merge(qualifying, on=["raceId", "driverId"], how="left")

    # Grid 0 means a pit-lane start: treat it as starting last.
    # A missing grid (rare) falls back to the qualifying position.
    df["grid"] = df["grid"].replace(0, NO_INFO_POS).astype(float)
    df["grid"] = df["grid"].fillna(df["quali_pos"]).fillna(NO_INFO_POS)
    df["quali_pos"] = df["quali_pos"].fillna(df["grid"])

    df = df.merge(
        _previous_standings(df, driver_st, "driverId", "driver"),
        on=["raceId", "driverId"],
        how="left",
    )
    df = df.merge(
        _previous_standings(df, team_st, "constructorId", "team"),
        on=["raceId", "constructorId"],
        how="left",
    )
    for p in ("driver", "team"):
        df[f"{p}_champ_pos"] = df[f"{p}_champ_pos"].fillna(NO_INFO_POS)
        df[f"{p}_points_share"] = df[f"{p}_points_share"].fillna(0.0)

    # Driver form over previous races (any season).
    df["driver_avg_finish_3"] = _rolling_past(df, "driverId", "positionOrder", 3, "mean")
    df["driver_podium_rate_5"] = _rolling_past(df, "driverId", "podium", 5, "mean")
    df["finished_f"] = df["finished"].astype(float)
    df["driver_finish_rate_10"] = _rolling_past(df, "driverId", "finished_f", 10, "mean")

    # Team form: the team's best finish in each of its previous 3 races.
    team_race = (
        df.groupby(["constructorId", "raceId", "date"], as_index=False)["positionOrder"]
        .min()
        .sort_values("date")
    )
    team_race["team_best_finish_3"] = _rolling_past(
        team_race, "constructorId", "positionOrder", 3, "mean"
    )
    df = df.merge(
        team_race[["constructorId", "raceId", "team_best_finish_3"]],
        on=["constructorId", "raceId"],
        how="left",
    )

    # Qualifying gap to teammate (negative = out-qualified the teammate).
    team_mean = df.groupby(["raceId", "constructorId"])["quali_pos"].transform("mean")
    team_n = df.groupby(["raceId", "constructorId"])["quali_pos"].transform("count")
    df["teammate_quali_gap"] = np.where(team_n > 1, (df["quali_pos"] - team_mean) * 2, 0.0)

    # Drivers with no history yet (rookies) get neutral-to-poor defaults.
    df["driver_avg_finish_3"] = df["driver_avg_finish_3"].fillna(15.0)
    df["driver_podium_rate_5"] = df["driver_podium_rate_5"].fillna(0.0)
    df["driver_finish_rate_10"] = df["driver_finish_rate_10"].fillna(0.8)
    df["team_best_finish_3"] = df["team_best_finish_3"].fillna(12.0)
    return df.drop(columns=["finished_f"])
