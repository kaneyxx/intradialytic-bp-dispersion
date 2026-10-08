from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass
class MedianMissingIndicatorPreprocessor:
    """Train-only median imputation with explicit missing indicators."""

    feature_cols: Sequence[str]
    add_missing_indicator: bool = True
    scale_numeric: bool = False

    def __post_init__(self) -> None:
        self.feature_cols = list(self.feature_cols)
        self.output_cols_: list[str] | None = None
        self.medians_: pd.Series | None = None
        self.means_: pd.Series | None = None
        self.scales_: pd.Series | None = None
        self.fitted_: bool = False

    def fit(self, frame: pd.DataFrame) -> "MedianMissingIndicatorPreprocessor":
        missing = [col for col in self.feature_cols if col not in frame.columns]
        if missing:
            raise KeyError(f"Missing feature columns for preprocessing: {missing}")
        values = frame.loc[:, self.feature_cols].apply(pd.to_numeric, errors="coerce").astype("float64")
        med = values.median(axis=0, skipna=True).fillna(0.0)
        self.medians_ = med.astype(float)
        filled = values.fillna(self.medians_).astype(float)
        self.means_ = filled.mean(axis=0).astype(float)
        std = filled.std(axis=0, skipna=True).replace(0.0, 1.0).fillna(1.0)
        self.scales_ = std.astype(float)
        output = list(self.feature_cols)
        if self.add_missing_indicator:
            output.extend(f"{col}_is_missing" for col in self.feature_cols)
        self.output_cols_ = output
        self.fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_ or self.medians_ is None or self.output_cols_ is None:
            raise RuntimeError("Preprocessor is not fitted.")
        values = frame.loc[:, self.feature_cols].apply(pd.to_numeric, errors="coerce").astype("float64")
        indicators = values.isna().astype(float)
        filled = values.fillna(self.medians_).astype(float)
        if self.scale_numeric:
            if self.means_ is None or self.scales_ is None:
                raise RuntimeError("Preprocessor scale statistics are not fitted.")
            filled = ((filled - self.means_) / self.scales_).astype(float)
        if not self.add_missing_indicator:
            return filled.loc[:, self.output_cols_]
        indicators.columns = [f"{col}_is_missing" for col in self.feature_cols]
        return pd.concat([filled, indicators], axis=1).loc[:, self.output_cols_]

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    @property
    def output_cols(self) -> list[str]:
        if self.output_cols_ is None:
            raise RuntimeError("Preprocessor is not fitted.")
        return list(self.output_cols_)


class ConstantSupportPreprocessor(MedianMissingIndicatorPreprocessor):
    """Median/indicator preprocessing with a support guard.

    A missing-value indicator that is constant in the fitting partition (the input was never, or always, missing there)
    has no separately learned effect; at application time it is held at the state observed in fitting, and the model
    evaluates the imputed numeric value. Numeric imputation and standardisation are unchanged.
    """

    frozen_missing_indicators_: dict = {}

    def transform(self, frame):
        out = super().transform(frame)
        for column, value in getattr(self, 'frozen_missing_indicators_', {}).items():
            out[column] = float(value)
        return out
