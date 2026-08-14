"""
Realistic test data generators for calibration testing.

This module provides functions to generate synthetic datasets that mimic
common miscalibration patterns observed in real machine learning models.
"""

import inspect
import zlib
from typing import Any

import numpy as np


class CalibrationDataGenerator:
    """Generate realistic calibration test data with various miscalibration patterns."""

    def __init__(self, random_state: int = 42):
        """Initialize the data generator.

        Parameters
        ----------
        random_state : int, default=42
            Random seed for reproducible results.
        """
        self.random_state = random_state
        self.random_state = random_state
        np.random.seed(random_state)

    def overconfident_neural_network(
        self, n_samples: int = 1000, noise_level: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate data mimicking overconfident neural network predictions.

        Neural networks tend to be overconfident, producing predictions
        clustered near 0 and 1 with few intermediate values.

        Parameters
        ----------
        n_samples : int, default=1000
            Number of samples to generate.
        noise_level : float, default=0.1
            Amount of noise to add to the true probabilities.

        Returns
        -------
        y_pred : ndarray
            Overconfident model predictions.
        y_true : ndarray
            True binary labels.
        """
        # Generate true probabilities with smooth distribution
        true_probs = np.random.beta(2, 2, n_samples)

        # Create overconfident predictions using Beta(0.5, 0.5) which concentrates at extremes
        overconfident_transform = np.random.beta(0.5, 0.5, n_samples)

        # Map smooth probabilities to overconfident ones
        y_pred = np.where(
            true_probs > 0.5,
            0.5 + 0.5 * overconfident_transform,
            0.5 * overconfident_transform,
        )

        # Add small amount of noise
        y_pred += np.random.normal(0, noise_level * 0.1, n_samples)
        y_pred = np.clip(y_pred, 0, 1)

        # Generate true labels
        y_true = np.random.binomial(1, true_probs, n_samples)

        return y_pred, y_true

    def underconfident_random_forest(
        self, n_samples: int = 1000, noise_level: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate data mimicking underconfident random forest predictions.

        Random forests tend to be underconfident, with predictions
        clustered around 0.5 due to averaging.

        Parameters
        ----------
        n_samples : int, default=1000
            Number of samples to generate.
        noise_level : float, default=0.1
            Amount of noise to add to the predictions.

        Returns
        -------
        y_pred : ndarray
            Underconfident model predictions.
        y_true : ndarray
            True binary labels.
        """
        # Generate true probabilities
        true_probs = np.random.uniform(0, 1, n_samples)

        # Create underconfident predictions using Beta(3, 3) which concentrates around 0.5
        underconfident_factor = np.random.beta(3, 3, n_samples)

        # Shrink predictions toward 0.5
        y_pred = 0.5 + (true_probs - 0.5) * underconfident_factor

        # Add noise
        y_pred += np.random.normal(0, noise_level, n_samples)
        y_pred = np.clip(y_pred, 0, 1)

        # Generate true labels
        y_true = np.random.binomial(1, true_probs, n_samples)

        return y_pred, y_true

    def sigmoid_temperature_distorted(
        self, n_samples: int = 1000, temperature: float = 2.0, noise_level: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate data with sigmoid temperature scaling distortion.

        Models trained with improper temperature scaling show
        systematic sigmoid-shaped calibration curves.

        Parameters
        ----------
        n_samples : int, default=1000
            Number of samples to generate.
        temperature : float, default=2.0
            Temperature parameter. > 1 makes model underconfident,
            < 1 makes it overconfident.
        noise_level : float, default=0.1
            Amount of noise to add.

        Returns
        -------
        y_pred : ndarray
            Temperature-distorted predictions.
        y_true : ndarray
            True binary labels.
        """
        # Generate logits from a normal distribution
        logits = np.random.normal(0, 2, n_samples)

        # Apply temperature scaling (higher temp = more conservative)
        temp_logits = logits / temperature

        # Convert to probabilities
        y_pred = 1 / (1 + np.exp(-temp_logits))

        # True probabilities without temperature distortion
        true_probs = 1 / (1 + np.exp(-logits))

        # Add noise
        y_pred += np.random.normal(0, noise_level, n_samples)
        y_pred = np.clip(y_pred, 0, 1)

        # Generate true labels
        y_true = np.random.binomial(1, true_probs, n_samples)

        return y_pred, y_true

    def imbalanced_binary_classification(
        self,
        n_samples: int = 1000,
        minority_ratio: float = 0.05,
        bias_strength: float = 0.3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate data from imbalanced classification with prediction bias.

        Imbalanced datasets often lead to biased probability estimates,
        especially after resampling techniques.

        Parameters
        ----------
        n_samples : int, default=1000
            Number of samples to generate.
        minority_ratio : float, default=0.05
            Fraction of positive class samples.
        bias_strength : float, default=0.3
            Strength of the bias toward majority class.

        Returns
        -------
        y_pred : ndarray
            Biased model predictions.
        y_true : ndarray
            True binary labels with class imbalance.
        """
        # Generate true labels with imbalance
        n_positive = int(n_samples * minority_ratio)
        y_true = np.concatenate([np.ones(n_positive), np.zeros(n_samples - n_positive)])

        # Shuffle
        shuffle_idx = np.random.permutation(n_samples)
        y_true = y_true[shuffle_idx]

        # Generate predictions with bias toward majority class
        y_pred = np.zeros(n_samples)

        # For positive samples: should be high but biased down
        pos_mask = y_true == 1
        y_pred[pos_mask] = (
            np.random.beta(2, 1, np.sum(pos_mask)) * (1 - bias_strength) + bias_strength
        )

        # For negative samples: should be low but sometimes biased up
        neg_mask = y_true == 0
        y_pred[neg_mask] = np.random.beta(1, 4, np.sum(neg_mask)) * bias_strength

        return y_pred, y_true

    def multi_modal_ensemble_disagreement(
        self, n_samples: int = 1000, n_modes: int = 3, noise_level: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate data with multi-modal prediction distribution.

        Ensemble models sometimes show multi-modal distributions
        when base models disagree on certain regions.

        Parameters
        ----------
        n_samples : int, default=1000
            Number of samples to generate.
        n_modes : int, default=3
            Number of modes in the distribution.
        noise_level : float, default=0.1
            Amount of noise to add.

        Returns
        -------
        y_pred : ndarray
            Multi-modal ensemble predictions.
        y_true : ndarray
            True binary labels.
        """
        # Create mixture of distributions
        mode_centers = np.linspace(0.1, 0.9, n_modes)
        mode_weights = np.random.dirichlet(np.ones(n_modes))

        y_pred = np.zeros(n_samples)
        y_true = np.zeros(n_samples, dtype=int)

        # Assign samples to modes
        mode_assignment = np.random.choice(n_modes, n_samples, p=mode_weights)

        for i in range(n_modes):
            mask = mode_assignment == i
            n_mode_samples = np.sum(mask)

            if n_mode_samples > 0:
                # Generate predictions around mode center
                center = mode_centers[i]
                width = 0.15  # Standard deviation

                mode_preds = np.random.normal(center, width, n_mode_samples)
                mode_preds = np.clip(mode_preds, 0, 1)

                y_pred[mask] = mode_preds

                # Generate true labels based on actual probabilities
                y_true[mask] = np.random.binomial(1, mode_preds, n_mode_samples)

        # Add noise
        y_pred += np.random.normal(0, noise_level, n_samples)
        y_pred = np.clip(y_pred, 0, 1)

        return y_pred, y_true

    def weather_forecasting_pattern(
        self,
        n_samples: int = 1000,
        temporal_correlation: float = 0.7,
        seasonal_amplitude: float = 0.3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate data mimicking weather forecasting patterns.

        Weather prediction has smooth probability gradients with
        temporal correlation and seasonal patterns.

        Parameters
        ----------
        n_samples : int, default=1000
            Number of samples to generate.
        temporal_correlation : float, default=0.7
            Strength of temporal correlation.
        seasonal_amplitude : float, default=0.3
            Amplitude of seasonal variation.

        Returns
        -------
        y_pred : ndarray
            Weather-like predictions.
        y_true : ndarray
            True binary labels (e.g., rain/no rain).
        """
        # Generate temporal indices
        time_indices = np.linspace(0, 4 * np.pi, n_samples)  # 2 full cycles

        # Seasonal base pattern
        seasonal_pattern = seasonal_amplitude * np.sin(time_indices)

        # Add temporal correlation using AR(1) process
        base_signal = np.zeros(n_samples)
        base_signal[0] = np.random.normal(0, 1)

        for i in range(1, n_samples):
            base_signal[i] = temporal_correlation * base_signal[i - 1] + np.sqrt(
                1 - temporal_correlation**2
            ) * np.random.normal(0, 1)

        # Combine patterns and convert to probabilities
        logits = base_signal + seasonal_pattern
        y_pred = 1 / (1 + np.exp(-logits))

        # Generate true labels
        y_true = np.random.binomial(1, y_pred, n_samples)

        return y_pred, y_true

    def click_through_rate_pattern(
        self,
        n_samples: int = 1000,
        power_law_alpha: float = 1.5,
        noise_level: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate data mimicking click-through rate prediction.

        CTR prediction often shows power-law distribution with
        heavy tail toward low probabilities.

        Parameters
        ----------
        n_samples : int, default=1000
            Number of samples to generate.
        power_law_alpha : float, default=1.5
            Power law exponent.
        noise_level : float, default=0.05
            Amount of noise to add.

        Returns
        -------
        y_pred : ndarray
            CTR-like predictions.
        y_true : ndarray
            True binary labels (click/no click).
        """
        # Generate power-law distributed base rates
        # Most predictions should be very low (typical CTR is 1-5%)
        uniform_samples = np.random.uniform(0.001, 1, n_samples)
        base_rates = np.power(uniform_samples, -1 / power_law_alpha)
        base_rates = np.clip(base_rates * 0.01, 0, 0.5)  # Scale to reasonable CTR range

        # Add some higher probability items (premium placements)
        high_prob_mask = np.random.random(n_samples) < 0.05  # 5% high-value items
        base_rates[high_prob_mask] += np.random.uniform(
            0.1, 0.4, np.sum(high_prob_mask)
        )
        base_rates = np.clip(base_rates, 0, 1)

        # Predictions are slightly miscalibrated
        y_pred = base_rates + np.random.normal(0, noise_level, n_samples)
        y_pred = np.clip(y_pred, 0, 1)

        # Generate true labels
        y_true = np.random.binomial(1, base_rates, n_samples)

        return y_pred, y_true

    def medical_diagnosis_pattern(
        self,
        n_samples: int = 1000,
        disease_prevalence: float = 0.01,
        test_sensitivity: float = 0.95,
        test_specificity: float = 0.98,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate data mimicking medical diagnosis scenarios.

        Medical diagnosis involves rare diseases with high-stakes
        decisions and specific sensitivity/specificity requirements.

        Parameters
        ----------
        n_samples : int, default=1000
            Number of samples to generate.
        disease_prevalence : float, default=0.01
            True prevalence of the disease (1%).
        test_sensitivity : float, default=0.95
            True positive rate.
        test_specificity : float, default=0.98
            True negative rate.

        Returns
        -------
        y_pred : ndarray
            Medical test predictions.
        y_true : ndarray
            True disease status.
        """
        # Generate true disease status
        y_true = np.random.binomial(1, disease_prevalence, n_samples)

        # Generate test results based on sensitivity/specificity
        y_pred = np.zeros(n_samples)

        # For diseased patients
        diseased_mask = y_true == 1
        n_diseased = np.sum(diseased_mask)
        if n_diseased > 0:
            # High confidence for true positives, some false negatives
            true_positives = np.random.random(n_diseased) < test_sensitivity
            y_pred[diseased_mask] = np.where(
                true_positives,
                np.random.beta(8, 2, n_diseased),  # High confidence
                np.random.beta(2, 8, n_diseased),  # Low confidence (false negatives)
            )

        # For healthy patients
        healthy_mask = y_true == 0
        n_healthy = np.sum(healthy_mask)
        if n_healthy > 0:
            # Low confidence for true negatives, some false positives
            true_negatives = np.random.random(n_healthy) < test_specificity
            y_pred[healthy_mask] = np.where(
                true_negatives,
                np.random.beta(2, 8, n_healthy),  # Low confidence
                np.random.beta(8, 2, n_healthy),  # High confidence (false positives)
            )

        return y_pred, y_true

    def generate_dataset(
        self, pattern: str, n_samples: int = 1000, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a dataset with the specified miscalibration pattern.

        Parameters
        ----------
        pattern : str
            Type of miscalibration pattern. One of:
            - 'overconfident_nn'
            - 'underconfident_rf'
            - 'sigmoid_distorted'
            - 'imbalanced_binary'
            - 'multi_modal'
            - 'weather_forecasting'
            - 'click_through_rate'
            - 'medical_diagnosis'
        n_samples : int, default=1000
            Number of samples to generate.
        **kwargs
            Additional parameters specific to each pattern.

        Returns
        -------
        y_pred : ndarray
            Model predictions (potentially miscalibrated).
        y_true : ndarray
            True binary labels.
        """
        # Reseed from a hash of the request. Every generator below draws from
        # the global np.random, so without this the data a test receives depends
        # on how many draws earlier tests made -- making the suite
        # order-dependent and non-reproducible under -k filters or xdist.
        np.random.seed(
            (
                self.random_state
                + zlib.crc32(
                    repr((pattern, n_samples, sorted(kwargs.items()))).encode()
                )
            )
            % (2**32)
        )
        generators = self._generators()

        if pattern not in generators:
            raise ValueError(
                f"Unknown pattern '{pattern}'. Available patterns: {list(generators.keys())}"
            )

        return generators[pattern](n_samples=n_samples, **kwargs)

    def _generators(self) -> dict[str, Any]:
        """Map each pattern name to the method that generates it.

        Returns
        -------
        dict
            Pattern name to bound generator method.
        """
        return {
            "overconfident_nn": self.overconfident_neural_network,
            "underconfident_rf": self.underconfident_random_forest,
            "sigmoid_distorted": self.sigmoid_temperature_distorted,
            "imbalanced_binary": self.imbalanced_binary_classification,
            "multi_modal": self.multi_modal_ensemble_disagreement,
            "weather_forecasting": self.weather_forecasting_pattern,
            "click_through_rate": self.click_through_rate_pattern,
            "medical_diagnosis": self.medical_diagnosis_pattern,
        }

    def accepts(self, pattern: str, parameter: str) -> bool:
        """Whether a pattern's generator takes the named keyword.

        Callers vary parameters that only some patterns support, and a
        hard-coded list of which is which drifts silently: it named
        ``click_through_rate`` as not taking ``noise_level`` when it does, and
        omitted ``imbalanced_binary``, which does not -- so every
        ``imbalanced_binary`` combination in the comprehensive matrix raised
        ``TypeError`` and was counted as a calibrator failure. Deriving the
        answer from the signature cannot drift.

        Parameters
        ----------
        pattern
            Pattern name.
        parameter
            Keyword to test for.

        Returns
        -------
        bool
            True if the generator accepts that keyword.

        Raises
        ------
        ValueError
            If the pattern is unknown.
        """
        generators = self._generators()
        if pattern not in generators:
            raise ValueError(
                f"Unknown pattern '{pattern}'. Available: {sorted(generators)}"
            )
        return parameter in inspect.signature(generators[pattern]).parameters
