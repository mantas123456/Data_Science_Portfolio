"""Load the Ergast-format Formula 1 tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def read(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / f"{name}.csv", na_values=["\\N"])


def load_results(first_year: int = 2010, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One row per driver per race, with the race date, round and finishing order.

    2010 is the default start because the current points system began that season.
    """
    races = read("races", data_dir)[["raceId", "year", "round", "circuitId", "name", "date"]]
    res = read("results", data_dir)[
        ["raceId", "driverId", "constructorId", "grid", "positionOrder", "statusId"]
    ]
    status = read("status", data_dir).set_index("statusId")["status"]
    df = res.merge(races, on="raceId")
    df = df[df["year"] >= first_year].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["finished"] = df["statusId"].map(status).str.match(r"^(Finished|\+\d+ Laps?)$")
    df["podium"] = (df["positionOrder"] <= 3).astype(int)
    return df.sort_values(["date", "positionOrder"]).reset_index(drop=True)


def load_qualifying(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    q = read("qualifying", data_dir)[["raceId", "driverId", "position"]]
    return (
        q.groupby(["raceId", "driverId"], as_index=False)["position"]
        .min()
        .rename(columns={"position": "quali_pos"})
    )


def load_standings(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Championship standings *after* each race (as published by Ergast)."""
    d = read("driver_standings", data_dir)[["raceId", "driverId", "points", "position"]]
    c = read("constructor_standings", data_dir)[["raceId", "constructorId", "points", "position"]]
    return d, c
