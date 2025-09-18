"""
Diagnostic tools for isotonic regression plateau analysis.

This module provides comprehensive diagnostics to distinguish between noise-based
flattening (good) and limited-data flattening (bad) in isotonic regression calibration.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import SplineTransformer

from .utils import (
    bootstrap_resample,
    check_arrays,
    compute_delong_ci,
    extract_plateaus,
    minimum_detectable_difference,
)

logger = logging.getLogger(__name__)


class PlateauInfo:
    """Information about a plateau in isotonic regression output."""

    def __init__(
        self,
        start_idx: int,
        end_idx: int,
        value: float,
        x_range: Tuple[float, float],
        sample_size: int,
    ):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.value = value
        self.x_range = x_range
        self.sample_size = sample_size
        self.width = end_idx - start_idx + 1

        # Diagnostic results (filled by analysis)
        self.tie_stability: Optional[float] = None
        self.conditional_auc: Optional[float] = None
        self.conditional_auc_ci: Optional[Tuple[float, float]] = None
        self.mdd_left: Optional[float] = None
        self.mdd_right: Optional[float] = None
        self.local_slope: Optional[float] = None
        self.local_slope_ci: Optional[Tuple[float, float]] = None
        self.classification: Optional[str] = None

    def __repr__(self):
        return (
            f"PlateauInfo(indices={self.start_idx}-{self.end_idx}, "
            f"value={self.value:.3f}, width={self.width}, "
            f"classification={self.classification})"
        )


class PlateauAnalyzer:
    """Helper class for analyzing individual plateaus."""

    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance

    def identify_plateaus(
        self, X: np.ndarray, y_calibrated: np.ndarray
    ) -> List[PlateauInfo]:
        """
        Identify plateaus in isotonic regression output.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Input features (should be sorted).
        y_calibrated : array-like of shape (n_samples,)
            Calibrated output values from isotonic regression.

        Returns
        -------
        plateaus : list of PlateauInfo
            List of identified plateaus.
        """
        X = np.asarray(X)
        y_calibrated = np.asarray(y_calibrated)

        plateau_tuples = extract_plateaus(X, y_calibrated, self.tolerance)
        plateaus = []

        for start_idx, end_idx, value in plateau_tuples:
            x_range = (X[start_idx], X[end_idx])
            sample_size = end_idx - start_idx + 1

            plateau = PlateauInfo(start_idx, end_idx, value, x_range, sample_size)
            plateaus.append(plateau)

        return plateaus

    def compute_mdd_for_plateau(
        self,
        plateau: PlateauInfo,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.05,
        power: float = 0.8,
    ) -> Tuple[float, float]:
        """
        Compute minimum detectable differences at plateau boundaries.

        Parameters
        ----------
        plateau : PlateauInfo
            Plateau to analyze.
        X : array-like of shape (n_samples,)
            Input features.
        y : array-like of shape (n_samples,)
            Target values.
        alpha : float, default=0.05
            Significance level.
        power : float, default=0.8
            Statistical power.

        Returns
        -------
        mdd_left : float
            MDD at left boundary.
        mdd_right : float
            MDD at right boundary.
        """
        # Get samples in plateau and adjacent regions
        plateau_y = y[plateau.start_idx : plateau.end_idx + 1]
        p_pooled = np.mean(plateau_y)
        n_plateau = len(plateau_y)

        # Left boundary
        if plateau.start_idx > 0:
            left_y = y[: plateau.start_idx]
            n_left = len(left_y)
            mdd_left = minimum_detectable_difference(
                n_left, n_plateau, p_pooled, alpha, power
            )
        else:
            mdd_left = np.inf

        # Right boundary
        if plateau.end_idx < len(y) - 1:
            right_y = y[plateau.end_idx + 1 :]
            n_right = len(right_y)
            mdd_right = minimum_detectable_difference(
                n_plateau, n_right, p_pooled, alpha, power
            )
        else:
            mdd_right = np.inf

        return mdd_left, mdd_right


class IsotonicDiagnostics:
    """
    Main class for diagnosing isotonic regression plateaus.

    This class provides comprehensive diagnostics to distinguish between
    noise-based flattening (good) and limited-data flattening (bad).

    Parameters
    ----------
    n_bootstraps : int, default=100
        Number of bootstrap samples for stability analysis.
    n_splits : int, default=5
        Number of cross-validation splits for stability analysis.
    alpha : float, default=0.05
        Significance level for statistical tests.
    power : float, default=0.8
        Statistical power for MDD calculations.
    random_state : int, optional
        Random state for reproducibility.

    Attributes
    ----------
    plateaus_ : list of PlateauInfo
        Identified plateaus from the last analysis.
    stability_results_ : dict
        Results from stability analysis.
    """

    def __init__(
        self,
        n_bootstraps: int = 100,
        n_splits: int = 5,
        alpha: float = 0.05,
        power: float = 0.8,
        random_state: Optional[int] = None,
    ):
        self.n_bootstraps = n_bootstraps
        self.n_splits = n_splits
        self.alpha = alpha
        self.power = power
        self.random_state = random_state
        self.plateau_analyzer = PlateauAnalyzer()

        self.plateaus_: Optional[List[PlateauInfo]] = None
        self.stability_results_: Optional[Dict] = None

    def analyze(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive plateau analysis.

        Parameters
        ----------
        X_train : array-like of shape (n_samples,)
            Training input features.
        y_train : array-like of shape (n_samples,)
            Training target values.
        X_test : array-like of shape (n_test_samples,), optional
            Test input features for conditional AUC analysis.
        y_test : array-like of shape (n_test_samples,), optional
            Test target values for conditional AUC analysis.

        Returns
        -------
        results : dict
            Comprehensive analysis results.
        """
        X_train, y_train = check_arrays(X_train, y_train)

        if X_test is not None and y_test is not None:
            X_test, y_test = check_arrays(X_test, y_test)

        # Fit isotonic regression on training data
        iso_reg = IsotonicRegression(out_of_bounds="clip")
        iso_reg.fit(X_train, y_train)
        y_calibrated = iso_reg.transform(X_train)

        # Identify plateaus
        self.plateaus_ = self.plateau_analyzer.identify_plateaus(X_train, y_calibrated)

        if not self.plateaus_:
            return {
                "n_plateaus": 0,
                "plateaus": [],
                "summary": "No plateaus detected in isotonic regression fit.",
            }

        # Perform diagnostic analyses
        self._analyze_tie_stability(X_train, y_train)
        self._analyze_conditional_auc(X_train, y_train, X_test, y_test)
        self._analyze_mdd(X_train, y_train)
        self._analyze_local_slopes(X_train, y_train)
        self._classify_plateaus()

        # Generate summary
        results = self._generate_summary()
        return results

    def _analyze_tie_stability(self, X: np.ndarray, y: np.ndarray) -> None:
        """Analyze tie stability across bootstrap resamples."""
        logger.info("Analyzing tie stability across bootstrap resamples...")

        rng = np.random.RandomState(self.random_state)

        for plateau in self.plateaus_:
            tie_counts = 0
            total_bootstraps = 0

            for i in range(self.n_bootstraps):
                # Bootstrap resample
                X_boot, y_boot = bootstrap_resample(
                    X, y, random_state=rng.randint(0, 2**31)
                )

                # Fit isotonic regression
                try:
                    iso_reg = IsotonicRegression(out_of_bounds="clip")
                    iso_reg.fit(X_boot, y_boot)
                    y_boot_cal = iso_reg.transform(X_boot)

                    # Check if original plateau region is still tied
                    # Map original indices to bootstrap space (approximate)
                    original_x_range = (X[plateau.start_idx], X[plateau.end_idx])
                    mask = (X_boot >= original_x_range[0]) & (
                        X_boot <= original_x_range[1]
                    )

                    if np.sum(mask) > 1:
                        boot_cal_in_range = y_boot_cal[mask]
                        # Check if values are approximately equal (forming a tie)
                        if np.std(boot_cal_in_range) < 1e-6:
                            tie_counts += 1

                    total_bootstraps += 1

                except Exception as e:
                    logger.warning(f"Bootstrap {i} failed: {e}")
                    continue

            if total_bootstraps > 0:
                plateau.tie_stability = tie_counts / total_bootstraps
            else:
                plateau.tie_stability = 0.0

    def _analyze_conditional_auc(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray],
        y_test: Optional[np.ndarray],
    ) -> None:
        """Analyze conditional AUC among tied pairs."""
        if X_test is None or y_test is None:
            logger.warning("Test data not provided, skipping conditional AUC analysis")
            for plateau in self.plateaus_:
                plateau.conditional_auc = None
                plateau.conditional_auc_ci = None
            return

        logger.info("Analyzing conditional AUC among tied pairs...")

        # Fit isotonic regression
        iso_reg = IsotonicRegression(out_of_bounds="clip")
        iso_reg.fit(X_train, y_train)

        for plateau in self.plateaus_:
            # Find test samples in plateau's X range
            x_min, x_max = plateau.x_range
            mask = (X_test >= x_min) & (X_test <= x_max)

            if np.sum(mask) < 2:
                plateau.conditional_auc = None
                plateau.conditional_auc_ci = None
                continue

            X_test_plateau = X_test[mask]
            y_test_plateau = y_test[mask]

            # Check if we have both classes
            if len(np.unique(y_test_plateau)) < 2:
                plateau.conditional_auc = None
                plateau.conditional_auc_ci = None
                continue

            # Compute AUC using original scores within plateau range
            auc, ci_lower, ci_upper = compute_delong_ci(
                y_test_plateau, X_test_plateau, self.alpha
            )

            plateau.conditional_auc = auc
            if not (np.isnan(ci_lower) or np.isnan(ci_upper)):
                plateau.conditional_auc_ci = (ci_lower, ci_upper)
            else:
                plateau.conditional_auc_ci = None

    def _analyze_mdd(self, X: np.ndarray, y: np.ndarray) -> None:
        """Analyze minimum detectable differences at plateau boundaries."""
        logger.info("Computing minimum detectable differences...")

        for plateau in self.plateaus_:
            mdd_left, mdd_right = self.plateau_analyzer.compute_mdd_for_plateau(
                plateau, X, y, self.alpha, self.power
            )
            plateau.mdd_left = mdd_left
            plateau.mdd_right = mdd_right

    def _analyze_local_slopes(self, X: np.ndarray, y: np.ndarray) -> None:
        """Analyze local slopes using smooth monotone fits."""
        logger.info("Analyzing local slopes with smooth monotone fits...")

        try:
            # Fit a monotone spline
            spline_transformer = SplineTransformer(
                n_splines=min(10, len(X) // 3), degree=3, include_bias=False
            )
            spline_transformer.fit(X.reshape(-1, 1))

            # Use isotonic regression on spline basis for monotonicity
            iso_spline = IsotonicRegression(out_of_bounds="clip")
            iso_spline.fit(X, y)
            y_smooth = iso_spline.transform(X)

            for plateau in self.plateaus_:
                # Estimate local slope in plateau region
                start_idx = max(0, plateau.start_idx - 1)
                end_idx = min(len(X) - 1, plateau.end_idx + 1)

                if end_idx > start_idx:
                    X_local = X[start_idx : end_idx + 1]
                    y_local = y_smooth[start_idx : end_idx + 1]

                    if len(X_local) > 1:
                        # Simple linear slope estimate
                        slope = (y_local[-1] - y_local[0]) / (X_local[-1] - X_local[0])

                        # Bootstrap confidence interval for slope
                        slopes = []
                        rng = np.random.RandomState(self.random_state)

                        for _ in range(min(50, self.n_bootstraps)):
                            indices = rng.choice(
                                len(X_local), size=len(X_local), replace=True
                            )
                            X_boot = X_local[indices]
                            y_boot = y_local[indices]

                            if len(np.unique(X_boot)) > 1:
                                sort_idx = np.argsort(X_boot)
                                X_boot_sorted = X_boot[sort_idx]
                                y_boot_sorted = y_boot[sort_idx]

                                boot_slope = (y_boot_sorted[-1] - y_boot_sorted[0]) / (
                                    X_boot_sorted[-1] - X_boot_sorted[0]
                                )
                                slopes.append(boot_slope)

                        plateau.local_slope = slope

                        if slopes:
                            ci_lower = np.percentile(slopes, 100 * self.alpha / 2)
                            ci_upper = np.percentile(slopes, 100 * (1 - self.alpha / 2))
                            plateau.local_slope_ci = (ci_lower, ci_upper)
                        else:
                            plateau.local_slope_ci = None
                    else:
                        plateau.local_slope = None
                        plateau.local_slope_ci = None
                else:
                    plateau.local_slope = None
                    plateau.local_slope_ci = None

        except Exception as e:
            logger.warning(f"Local slope analysis failed: {e}")
            for plateau in self.plateaus_:
                plateau.local_slope = None
                plateau.local_slope_ci = None

    def _classify_plateaus(self) -> None:
        """Classify plateaus as supported, limited-data, or inconclusive."""
        for plateau in self.plateaus_:
            criteria = self._extract_plateau_criteria(plateau)
            plateau.classification = self._determine_classification(criteria)

    def _extract_plateau_criteria(self, plateau) -> List[str]:
        """Extract diagnostic criteria for a plateau."""
        criteria = []

        # Tie stability criterion
        if plateau.tie_stability is not None:
            if plateau.tie_stability > 0.7:
                criteria.append("stable")
            elif plateau.tie_stability < 0.3:
                criteria.append("unstable")

        # Conditional AUC criterion
        if plateau.conditional_auc is not None:
            if plateau.conditional_auc < 0.55:
                criteria.append("low_auc")
            elif plateau.conditional_auc > 0.65:
                criteria.append("high_auc")

        # Local slope criterion
        if plateau.local_slope is not None and plateau.local_slope_ci is not None:
            ci_lower, ci_upper = plateau.local_slope_ci
            if ci_lower <= 0 <= ci_upper:
                criteria.append("flat_slope")
            elif ci_lower > 0:
                criteria.append("positive_slope")

        return criteria

    def _determine_classification(self, criteria: List[str]) -> str:
        """Determine plateau classification based on criteria."""
        supported_criteria = {"stable", "low_auc", "flat_slope"}
        limited_data_criteria = {"unstable", "high_auc", "positive_slope"}

        if supported_criteria.issubset(set(criteria)):
            return "supported"
        elif limited_data_criteria.issubset(set(criteria)):
            return "limited_data"
        else:
            return "inconclusive"

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary of results."""
        summary = {"n_plateaus": len(self.plateaus_), "plateaus": []}

        for i, plateau in enumerate(self.plateaus_):
            plateau_summary = {
                "plateau_id": i,
                "indices": (plateau.start_idx, plateau.end_idx),
                "x_range": plateau.x_range,
                "value": plateau.value,
                "width": plateau.width,
                "sample_size": plateau.sample_size,
                "tie_stability": plateau.tie_stability,
                "conditional_auc": plateau.conditional_auc,
                "conditional_auc_ci": plateau.conditional_auc_ci,
                "mdd_left": plateau.mdd_left,
                "mdd_right": plateau.mdd_right,
                "local_slope": plateau.local_slope,
                "local_slope_ci": plateau.local_slope_ci,
                "classification": plateau.classification,
            }
            summary["plateaus"].append(plateau_summary)

        # Overall statistics
        classifications = [p.classification for p in self.plateaus_ if p.classification]
        summary["classification_counts"] = {
            "supported": classifications.count("supported"),
            "limited_data": classifications.count("limited_data"),
            "inconclusive": classifications.count("inconclusive"),
        }

        return summary

    def plateau_summary(self) -> str:
        """Generate a human-readable summary of plateau analysis."""
        if not self.plateaus_:
            return "No plateaus detected."

        lines = [f"Detected {len(self.plateaus_)} plateau(s):"]
        lines.append("")

        for i, plateau in enumerate(self.plateaus_):
            lines.append(f"Plateau {i+1}:")
            lines.append(
                f"  Range: [{plateau.x_range[0]:.3f}, {plateau.x_range[1]:.3f}]"
            )
            lines.append(f"  Value: {plateau.value:.3f}")
            lines.append(f"  Width: {plateau.width} samples")
            lines.append(f"  Classification: {plateau.classification}")

            if plateau.tie_stability is not None:
                lines.append(f"  Tie stability: {plateau.tie_stability:.3f}")

            if plateau.conditional_auc is not None:
                auc_str = f"{plateau.conditional_auc:.3f}"
                if plateau.conditional_auc_ci is not None:
                    ci_lower, ci_upper = plateau.conditional_auc_ci
                    auc_str += f" (CI: [{ci_lower:.3f}, {ci_upper:.3f}])"
                lines.append(f"  Conditional AUC: {auc_str}")

            lines.append("")

        return "\n".join(lines)


def analyze_plateaus(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function for comprehensive plateau analysis.

    Parameters
    ----------
    X_train : array-like of shape (n_samples,)
        Training input features.
    y_train : array-like of shape (n_samples,)
        Training target values.
    X_test : array-like of shape (n_test_samples,), optional
        Test input features.
    y_test : array-like of shape (n_test_samples,), optional
        Test target values.
    **kwargs
        Additional arguments passed to IsotonicDiagnostics.

    Returns
    -------
    results : dict
        Comprehensive analysis results.
    """
    diagnostics = IsotonicDiagnostics(**kwargs)
    return diagnostics.analyze(X_train, y_train, X_test, y_test)
