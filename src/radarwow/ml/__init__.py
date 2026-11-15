"""radar-wow stage 9: machine learning (RainForest-style gauge-trained correction, proof of concept)."""

from .forest import FEATURES, RainForestQPE, gauge_samples, leave_one_case_out

PLANNED = ["DL non-meteorological echo QC", "nowcasting (pysteps / DL)"]
__all__ = ["FEATURES", "RainForestQPE", "gauge_samples", "leave_one_case_out"]
