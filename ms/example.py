"""Fit a calibration map and evaluate on separate simulated observations."""

import numpy as np

from calibre import CenteredIsotonicCalibrator, calibration_report

rng = np.random.default_rng(42)
latent = rng.normal(size=2000)
probability = 1 / (1 + np.exp(-latent))
scores = 1 / (1 + np.exp(-1.8 * latent))
y = rng.binomial(1, probability)

calibrator = CenteredIsotonicCalibrator().fit(scores[:1000], y[:1000])
predictions = calibrator.transform(scores[1000:])
report = calibration_report(y[1000:], predictions)
print(report)
