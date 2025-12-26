Advanced Usage Examples
=======================

This section covers advanced calibration techniques and specialized use cases.

Multi-Class Calibration
-----------------------

.. note::
   **Important**: Calibre is designed for binary calibration. This example shows how to extend binary calibration methods to multi-class problems using a One-vs-Rest approach. This is a user implementation pattern, not native multi-class support.

While Calibre focuses on binary calibration, you can extend it to multi-class problems using a One-vs-Rest strategy:

.. code-block:: python

   import numpy as np
   from sklearn.datasets import make_classification
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from calibre import NearlyIsotonicCalibrator
   
   # Generate multi-class dataset
   X, y = make_classification(
       n_samples=2000,
       n_features=20,
       n_classes=3,
       n_redundant=2,
       n_informative=18,
       random_state=42
   )
   
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.5, random_state=42
   )
   
   # Train model
   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)
   y_pred_proba = model.predict_proba(X_test)
   
   # Calibrate each class separately using One-vs-Rest approach
   # NOTE: This is a user implementation - not native Calibre functionality
   calibrators = {}
   y_pred_cal = np.zeros_like(y_pred_proba)
   
   for class_idx in range(3):
       # Create binary labels for current class
       y_binary = (y_test == class_idx).astype(int)
       y_pred_binary = y_pred_proba[:, class_idx]
       
       # Fit calibrator for this class
       calibrator = NearlyIsotonicCalibrator(lam=1.0, method='path')
       calibrator.fit(y_pred_binary, y_binary)
       
       # Store calibrator and get calibrated predictions
       calibrators[class_idx] = calibrator
       y_pred_cal[:, class_idx] = calibrator.transform(y_pred_binary)
   
   # Renormalize probabilities to sum to 1
   y_pred_cal = y_pred_cal / y_pred_cal.sum(axis=1, keepdims=True)
   
   print("Multi-class calibration completed")
   print(f"Original probability sums: {y_pred_proba.sum(axis=1)[:5]}")
   print(f"Calibrated probability sums: {y_pred_cal.sum(axis=1)[:5]}")

Custom Calibration Pipelines
----------------------------

Creating Calibration Ensembles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example shows how to create a custom ensemble class using Calibre's calibrators:

.. code-block:: python

   from calibre import (
       NearlyIsotonicCalibrator,
       SplineCalibrator,
       RelaxedPAVACalibrator,
       mean_calibration_error
   )
   
   # Custom ensemble class - not part of Calibre library
   class CalibrationEnsemble:
       """Ensemble of calibration methods."""
       
       def __init__(self, calibrators, weights=None):
           self.calibrators = calibrators
           self.weights = weights or [1.0] * len(calibrators)
           self.weights = np.array(self.weights) / np.sum(self.weights)
           
       def fit(self, X, y):
           """Fit all calibrators."""
           for calibrator in self.calibrators:
               calibrator.fit(X, y)
           return self
           
       def transform(self, X):
           """Ensemble prediction using weighted average."""
           predictions = []
           for calibrator in self.calibrators:
               pred = calibrator.transform(X)
               predictions.append(pred)
           
           # Weighted average
           ensemble_pred = np.zeros_like(predictions[0])
           for pred, weight in zip(predictions, self.weights):
               ensemble_pred += weight * pred
               
           return ensemble_pred
   
   # Create ensemble
   calibrators = [
       NearlyIsotonicCalibrator(lam=1.0, method='path'),
       SplineCalibrator(n_splines=10, degree=3, cv=3),
       RelaxedPAVACalibrator(percentile=10, adaptive=True)
   ]
   
   ensemble = CalibrationEnsemble(calibrators, weights=[0.4, 0.3, 0.3])
   ensemble.fit(y_pred_uncal, y_test)
   y_pred_ensemble = ensemble.transform(y_pred_uncal)
   
   print(f"Ensemble MCE: {mean_calibration_error(y_test, y_pred_ensemble):.4f}")

Adaptive Calibration Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.model_selection import validation_curve
   
   def select_best_calibrator(X, y, calibrators, cv=5):
       """Select best calibrator using cross-validation."""
       best_score = float('inf')
       best_calibrator = None
       best_name = None
       
       for name, calibrator in calibrators.items():
           # Use validation curve to estimate performance
           try:
               # Create parameter grid (using lambda for Nearly Isotonic as example)
               if hasattr(calibrator, 'lam'):
                   param_name = 'lam'
                   param_range = [0.1, 1.0, 10.0]
               else:
                   param_name = None
                   param_range = [None]
               
               scores = []
               for _ in range(cv):
                   # Simple holdout validation
                   X_train, X_val, y_train, y_val = train_test_split(
                       X, y, test_size=0.2, random_state=np.random.randint(1000)
                   )
                   
                   calibrator.fit(X_train, y_train)
                   y_pred = calibrator.transform(X_val)
                   score = mean_calibration_error(y_val, y_pred)
                   scores.append(score)
               
               avg_score = np.mean(scores)
               print(f"{name}: MCE = {avg_score:.4f} ± {np.std(scores):.4f}")
               
               if avg_score < best_score:
                   best_score = avg_score
                   best_calibrator = calibrator
                   best_name = name
                   
           except Exception as e:
               print(f"{name}: Failed - {e}")
       
       return best_calibrator, best_name, best_score
   
   # Test different calibrators
   calibrators = {
       'Nearly Isotonic (strict)': NearlyIsotonicCalibrator(lam=10.0),
       'Nearly Isotonic (moderate)': NearlyIsotonicCalibrator(lam=1.0),
       'Nearly Isotonic (relaxed)': NearlyIsotonicCalibrator(lam=0.1),
       'I-Spline': SplineCalibrator(n_splines=10),
       'Relaxed PAVA': RelaxedPAVACalibrator(percentile=10)
   }
   
   best_cal, best_name, best_score = select_best_calibrator(
       y_pred_uncal, y_test, calibrators
   )
   
   print(f"\\nBest calibrator: {best_name} (MCE: {best_score:.4f})")

Temperature Scaling Integration
------------------------------

Combining with temperature scaling for neural networks:

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   from sklearn.model_selection import train_test_split
   
   class TemperatureScaling(nn.Module):
       """Temperature scaling for neural network calibration."""
       
       def __init__(self):
           super().__init__()
           self.temperature = nn.Parameter(torch.ones(1))
           
       def forward(self, logits):
           return logits / self.temperature
   
   def temperature_scale_then_isotonic(logits, y_true, test_logits):
       """Apply temperature scaling followed by isotonic calibration."""
       
       # Convert to torch tensors
       logits_tensor = torch.FloatTensor(logits.reshape(-1, 1))
       y_tensor = torch.LongTensor(y_true)
       
       # Temperature scaling
       temp_model = TemperatureScaling()
       optimizer = optim.LBFGS([temp_model.temperature], lr=0.01, max_iter=50)
       
       def eval_loss():
           optimizer.zero_grad()
           scaled_logits = temp_model(logits_tensor)
           loss = nn.CrossEntropyLoss()(
               torch.cat([1-torch.sigmoid(scaled_logits), torch.sigmoid(scaled_logits)], 1),
               y_tensor
           )
           loss.backward()
           return loss
       
       optimizer.step(eval_loss)
       
       # Apply temperature scaling to test data
       test_logits_tensor = torch.FloatTensor(test_logits.reshape(-1, 1))
       with torch.no_grad():
           temp_scaled = torch.sigmoid(temp_model(test_logits_tensor)).numpy().ravel()
       
       # Apply isotonic calibration on top of temperature scaling
       calibrator = NearlyIsotonicCalibrator(lam=1.0)
       
       # Fit on temperature-scaled training predictions
       train_temp_scaled = torch.sigmoid(temp_model(logits_tensor)).detach().numpy().ravel()
       calibrator.fit(train_temp_scaled, y_true)
       
       # Final calibrated predictions
       final_calibrated = calibrator.transform(temp_scaled)
       
       return final_calibrated, temp_model.temperature.item()
   
   # Example usage (with synthetic logits)
   np.random.seed(42)
   logits_train = np.random.normal(0, 2, 1000)
   y_train_temp = (logits_train > 0).astype(int)
   logits_test = np.random.normal(0, 2, 500)
   
   y_final, optimal_temp = temperature_scale_then_isotonic(
       logits_train, y_train_temp, logits_test
   )
   
   print(f"Optimal temperature: {optimal_temp:.3f}")

Handling Concept Drift
----------------------

Adaptive calibration for changing data distributions:

.. code-block:: python

   from collections import deque
   
   class AdaptiveCalibrator:
       """Calibrator that adapts to concept drift."""
       
       def __init__(self, base_calibrator, window_size=1000, retrain_threshold=0.05):
           self.base_calibrator = base_calibrator
           self.window_size = window_size
           self.retrain_threshold = retrain_threshold
           
           self.prediction_buffer = deque(maxlen=window_size)
           self.target_buffer = deque(maxlen=window_size)
           self.calibration_error_history = deque(maxlen=100)
           
           self.is_fitted = False
           
       def update(self, y_pred, y_true):
           """Update with new prediction and true label."""
           self.prediction_buffer.append(y_pred)
           self.target_buffer.append(y_true)
           
           # Calculate recent calibration error
           if len(self.prediction_buffer) >= 50:  # Minimum samples for evaluation
               recent_error = mean_calibration_error(
                   list(self.target_buffer)[-50:],
                   list(self.prediction_buffer)[-50:]
               )
               self.calibration_error_history.append(recent_error)
               
               # Check if retraining is needed
               if len(self.calibration_error_history) >= 10:
                   recent_avg = np.mean(list(self.calibration_error_history)[-10:])
                   if len(self.calibration_error_history) >= 20:
                       older_avg = np.mean(list(self.calibration_error_history)[-20:-10])
                       
                       if recent_avg > older_avg + self.retrain_threshold:
                           self._retrain()
                           print(f"Retrained calibrator: error increased from {older_avg:.4f} to {recent_avg:.4f}")
           
       def _retrain(self):
           """Retrain calibrator on recent data."""
           if len(self.prediction_buffer) >= 100:
               X_recent = np.array(list(self.prediction_buffer))
               y_recent = np.array(list(self.target_buffer))
               
               # Create new calibrator instance
               from copy import deepcopy
               self.base_calibrator = deepcopy(self.base_calibrator)
               self.base_calibrator.fit(X_recent, y_recent)
               self.is_fitted = True
       
       def fit(self, X, y):
           """Initial fit."""
           self.base_calibrator.fit(X, y)
           self.is_fitted = True
           
           # Initialize buffers
           for x, y_val in zip(X, y):
               self.prediction_buffer.append(x)
               self.target_buffer.append(y_val)
               
           return self
       
       def transform(self, X):
           """Transform predictions."""
           if not self.is_fitted:
               raise ValueError("Calibrator not fitted")
           return self.base_calibrator.transform(X)
   
   # Example usage
   adaptive_cal = AdaptiveCalibrator(
       NearlyIsotonicCalibrator(lam=1.0),
       window_size=500,
       retrain_threshold=0.02
   )
   
   # Initial fit
   adaptive_cal.fit(y_pred_uncal[:500], y_test[:500])
   
   # Simulate streaming predictions with concept drift
   for i in range(500, len(y_pred_uncal), 10):
       batch_pred = y_pred_uncal[i:i+10]
       batch_true = y_test[i:i+10]
       
       # Get calibrated predictions
       batch_cal = adaptive_cal.transform(batch_pred)
       
       # Update with true labels (in practice, these come later)
       for pred, true in zip(batch_pred, batch_true):
           adaptive_cal.update(pred, true)

Calibration for Specific Domains
--------------------------------

Time Series Calibration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def time_series_calibration(y_pred, y_true, timestamps, window_days=30):
       """Time-aware calibration that uses recent data for fitting."""
       from datetime import datetime, timedelta
       
       # Convert timestamps to datetime if needed
       if isinstance(timestamps[0], str):
           timestamps = [datetime.fromisoformat(ts) for ts in timestamps]
       
       calibrated_predictions = np.zeros_like(y_pred)
       
       for i, current_time in enumerate(timestamps):
           # Define time window
           window_start = current_time - timedelta(days=window_days)
           
           # Find data within time window before current prediction
           mask = [(ts >= window_start) and (ts < current_time) for ts in timestamps]
           
           if np.sum(mask) >= 50:  # Minimum samples for calibration
               # Fit calibrator on recent data
               X_recent = y_pred[mask]
               y_recent = y_true[mask]
               
               calibrator = NearlyIsotonicCalibrator(lam=1.0)
               calibrator.fit(X_recent, y_recent)
               
               # Calibrate current prediction
               calibrated_predictions[i] = calibrator.transform([y_pred[i]])[0]
           else:
               # Not enough recent data, use uncalibrated prediction
               calibrated_predictions[i] = y_pred[i]
       
       return calibrated_predictions
   
   # Example with synthetic time series data
   from datetime import datetime, timedelta
   
   # Generate timestamps
   start_date = datetime(2024, 1, 1)
   timestamps = [start_date + timedelta(days=i) for i in range(len(y_pred_uncal))]
   
   # Apply time-series calibration
   y_pred_ts_cal = time_series_calibration(
       y_pred_uncal, y_test, timestamps, window_days=30
   )
   
   print(f"Time-series calibrated MCE: {mean_calibration_error(y_test, y_pred_ts_cal):.4f}")

High-Stakes Decision Making
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def conservative_calibration(y_pred, y_true, risk_tolerance=0.05):
       """Conservative calibration that errs on the side of caution."""
       
       # Use stricter calibration for high-stakes scenarios
       calibrator = NearlyIsotonicCalibrator(lam=50.0, method='cvx')  # Very strict
       calibrator.fit(y_pred, y_true)
       y_cal = calibrator.transform(y_pred)
       
       # Apply additional conservative adjustment
       # Push probabilities away from decision boundaries
       decision_threshold = 0.5
       adjustment_strength = risk_tolerance
       
       conservative_cal = y_cal.copy()
       
       # Make predictions more conservative (further from 0.5)
       above_threshold = y_cal > decision_threshold
       below_threshold = y_cal <= decision_threshold
       
       conservative_cal[above_threshold] = np.minimum(
           1.0,
           y_cal[above_threshold] + adjustment_strength
       )
       conservative_cal[below_threshold] = np.maximum(
           0.0,
           y_cal[below_threshold] - adjustment_strength
       )
       
       return conservative_cal
   
   # Apply conservative calibration
   y_pred_conservative = conservative_calibration(y_pred_uncal, y_test)
   
   print(f"Conservative calibration MCE: {mean_calibration_error(y_test, y_pred_conservative):.4f}")
   print(f"Mean prediction shift: {np.mean(np.abs(y_pred_conservative - y_pred_uncal)):.4f}")

Performance Optimization
------------------------

Efficient Batch Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def batch_calibration(model, calibrator, X_large, batch_size=10000):
       """Efficiently calibrate predictions for large datasets."""
       
       n_samples = len(X_large)
       n_batches = (n_samples + batch_size - 1) // batch_size
       
       calibrated_predictions = []
       
       for i in range(n_batches):
           start_idx = i * batch_size
           end_idx = min((i + 1) * batch_size, n_samples)
           
           # Get batch predictions
           X_batch = X_large[start_idx:end_idx]
           y_pred_batch = model.predict_proba(X_batch)[:, 1]
           
           # Calibrate batch
           y_cal_batch = calibrator.transform(y_pred_batch)
           calibrated_predictions.append(y_cal_batch)
           
           if (i + 1) % 10 == 0:
               print(f"Processed {i + 1}/{n_batches} batches")
       
       return np.concatenate(calibrated_predictions)
   
   # Example with large synthetic dataset
   np.random.seed(42)
   X_large = np.random.randn(50000, 20)
   
   # Assuming model and calibrator are already fitted
   y_pred_large_cal = batch_calibration(model, calibrator, X_large)