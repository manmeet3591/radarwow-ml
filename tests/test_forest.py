import numpy as np
import pandas as pd

from radarwow.ml import FEATURES, RainForestQPE, leave_one_case_out


def _cases(n=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for case, factor in [("a", 0.5), ("b", 0.6), ("c", 0.55)]:
        r = rng.gamma(1.0, 2.0, n)
        rows.append(pd.DataFrame({"case": case, "radar_mm": r, "radar_max": r * 1.5, "radar_std": r * 0.2,
                                  "beam_height_km": rng.uniform(0, 2, n), "blockage": 0.0, "distance_km": rng.uniform(0, 150, n),
                                  "c_band": 0.0, "gauge_mm": r / factor * np.exp(rng.normal(0, 0.2, n))}))
    return pd.concat(rows, ignore_index=True)


def test_forest_learns_a_shared_bias():
    out = leave_one_case_out(_cases())
    for case, d in out.groupby("case"):
        assert abs(d.forest.sum() / d.gauge_mm.sum() - 1) < 0.25
        assert abs(d.radar_mm.sum() / d.gauge_mm.sum() - 1) > 0.35


def test_importance_has_all_features():
    m = RainForestQPE(n_estimators=20).fit(_cases())
    assert set(m.importance().index) == set(FEATURES)
