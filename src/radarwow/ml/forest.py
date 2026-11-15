"""A RainForest-style correction of radar rain with gauges: a random forest learns hourly gauge rain from features of
the radar product around the gauge. After Wolfensberger, Gabella, Boscacci, Germann & Berne 2021, Atmos. Meas. Tech. 14,
3169-3193, doi:10.5194/amt-14-3169-2021 (RainForest, MeteoSwiss), who trained on polarimetric features at several
heights; this proof of concept uses what radarwow.pipeline.hourly_totals keeps: the hourly radar total and its spread
around the gauge, beam height, blockage, distance from the radar and band.

The target is log(1 + gauge mm), so the forest is not dominated by the few heaviest hours; predictions are transformed
back. Evaluate across events and radars (leave one case out): within one event, nearby gauge-hours are strongly
correlated and an in-sample score says little about a new radar or climate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["radar_mm", "radar_max", "radar_std", "beam_height_km", "blockage", "distance_km", "c_band"]


def gauge_samples(rad, gauges: pd.DataFrame, case: str) -> pd.DataFrame:
    """One row per gauge-hour: features from an `hourly_totals` Dataset around each gauge, plus the gauge value.

    ``gauges`` needs columns station, lat, lon, time, gauge_mm (as from radarwow.verify.fetch_*)."""
    lat0, lon0 = rad.attrs["radar_lat"], rad.attrs["radar_lon"]
    x, y = rad["x"].values, rad["y"].values
    dx, dy = x[1] - x[0], y[1] - y[0]
    g = gauges.copy()
    gx = (g["lon"].values - lon0) * 111.2 * np.cos(np.deg2rad(lat0))
    gy = (g["lat"].values - lat0) * 111.2
    j = np.round((gx - x[0]) / dx).astype(int)
    i = np.round((gy - y[0]) / dy).astype(int)
    inside = (i >= 2) & (i < y.size - 2) & (j >= 2) & (j < x.size - 2)
    h, b = rad["HEIGHT"].values, rad["BLOCKAGE"].values
    times = pd.DatetimeIndex(rad["time"].values)
    rows = []
    for k in np.flatnonzero(inside):
        t = pd.Timestamp(g["time"].iloc[k])
        if t not in times:
            continue
        f = rad["RAIN"].sel(time=t).values
        w3 = f[i[k] - 1:i[k] + 2, j[k] - 1:j[k] + 2]
        w5 = f[i[k] - 2:i[k] + 3, j[k] - 2:j[k] + 3]
        if not np.isfinite(w3).any():
            continue
        rows.append(dict(case=case, station=g["station"].iloc[k], time=t, gauge_mm=float(g["gauge_mm"].iloc[k]),
                         radar_mm=float(np.nanmean(w3)), radar_max=float(np.nanmax(w5)), radar_std=float(np.nanstd(w5)),
                         beam_height_km=float(h[i[k], j[k]]), blockage=float(np.nan_to_num(b[i[k], j[k]])),
                         distance_km=float(np.hypot(gx[k], gy[k])), c_band=float(rad.attrs.get("band", "S") == "C")))
    return pd.DataFrame(rows)


class RainForestQPE:
    """Random forest on log(1 + mm). ``fit(df)`` / ``predict(df)`` on frames from `gauge_samples`."""

    def __init__(self, n_estimators: int = 300, min_samples_leaf: int = 5, random_state: int = 0, features=FEATURES):
        from sklearn.ensemble import RandomForestRegressor

        self.features = list(features)
        self.model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=random_state, n_jobs=-1)

    def fit(self, df: pd.DataFrame):
        X = df[self.features].fillna(-1).values
        self.model.fit(X, np.log1p(df["gauge_mm"].values))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.expm1(self.model.predict(df[self.features].fillna(-1).values)).clip(0)

    def importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=self.features).sort_values(ascending=False)


def leave_one_case_out(df: pd.DataFrame, **kw) -> pd.DataFrame:
    """For each case: train on the others, predict it. Adds columns ``forest`` and ``scaled`` (radar times the
    training cases' total gauge / radar ratio, the simplest learned correction to beat)."""
    out = []
    for case in df["case"].unique():
        train, test = df[df.case != case], df[df.case == case].copy()
        test["forest"] = RainForestQPE(**kw).fit(train).predict(test)
        test["scaled"] = test["radar_mm"] * train["gauge_mm"].sum() / max(train["radar_mm"].sum(), 1e-9)
        out.append(test)
    return pd.concat(out, ignore_index=True)
