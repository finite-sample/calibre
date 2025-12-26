Basic Usage Examples
====================

This section provides step-by-step examples for common calibration tasks.

Complete Workflow Example
--------------------------

Here's a complete example showing how to train a model, calibrate its predictions, and evaluate the results:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.datasets import make_classification
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.calibration import calibration_curve
   
   from calibre import (
       NearlyIsotonicCalibrator,
       mean_calibration_error,
       expected_calibration_error,
       unique_value_counts
   )
   
   # Generate synthetic dataset
   X, y = make_classification(
       n_samples=2000, 
       n_features=20, 
       n_redundant=2, 
       n_informative=18,
       random_state=42
   )
   
   # Split data
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.5, random_state=42
   )
   
   # Train a Random Forest model
   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)
   
   # Get uncalibrated predictions
   y_pred_uncal = model.predict_proba(X_test)[:, 1]
   
   # Apply calibration
   calibrator = NearlyIsotonicCalibrator(lam=1.0, method='path')
   calibrator.fit(y_pred_uncal, y_test)
   y_pred_cal = calibrator.transform(y_pred_uncal)
   
   # Evaluate calibration quality
   mce_before = mean_calibration_error(y_test, y_pred_uncal)
   mce_after = mean_calibration_error(y_test, y_pred_cal)
   
   ece_before = expected_calibration_error(y_test, y_pred_uncal, n_bins=10)
   ece_after = expected_calibration_error(y_test, y_pred_cal, n_bins=10)
   
   # Check granularity preservation
   counts_before = unique_value_counts(y_pred_uncal)
   counts_after = unique_value_counts(y_pred_cal, y_orig=y_pred_uncal)
   
   print("Calibration Results:")
   print(f"Mean Calibration Error: {mce_before:.4f} → {mce_after:.4f}")
   print(f"Expected Calibration Error: {ece_before:.4f} → {ece_after:.4f}")
   print(f"Unique values: {counts_before['n_unique_y_pred']} → {counts_after['n_unique_y_pred']}")
   print(f"Preservation ratio: {counts_after['unique_value_ratio']:.3f}")

**For comprehensive method comparisons, see the Performance Comparison notebook.**

Handling Different Data Types
-----------------------------

Working with Imbalanced Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.datasets import make_classification
   
   # Create imbalanced dataset
   X_imbal, y_imbal = make_classification(
       n_samples=2000,
       n_features=20,
       weights=[0.9, 0.1],  # 90% class 0, 10% class 1
       random_state=42
   )
   
   X_train, X_test, y_train, y_test = train_test_split(
       X_imbal, y_imbal, test_size=0.5, stratify=y_imbal, random_state=42
   )
   
   # Train model
   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)
   y_pred = model.predict_proba(X_test)[:, 1]
   
   # Calibrate with method suitable for imbalanced data
   calibrator = RelaxedPAVACalibrator(percentile=5, adaptive=True)  # Lower percentile for imbalanced data
   calibrator.fit(y_pred, y_test)
   y_cal = calibrator.transform(y_pred)
   
   print(f"Class distribution: {np.bincount(y_test)}")
   print(f"MCE before: {mean_calibration_error(y_test, y_pred):.4f}")
   print(f"MCE after: {mean_calibration_error(y_test, y_cal):.4f}")

Working with Small Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Simulate small dataset
   np.random.seed(42)
   n_small = 200
   X_small = np.random.uniform(0, 1, n_small)
   y_small = np.random.binomial(1, X_small, n_small)
   
   # Use methods that work well with small datasets
   calibrators_small = {
       'I-Spline (small)': SplineCalibrator(n_splines=5, degree=2, cv=3),
       'Relaxed PAVA': RelaxedPAVACalibrator(percentile=20, adaptive=False),
       'Regularized': RegularizedIsotonicRegression(alpha=1.0)  # Higher regularization
   }
   
   for name, cal in calibrators_small.items():
       try:
           cal.fit(X_small, y_small)
           y_cal = cal.transform(X_small)
           mce = mean_calibration_error(y_small, y_cal)
           print(f"{name}: MCE = {mce:.4f}")
       except Exception as e:
           print(f"{name}: Failed - {e}")

Visualization Examples
----------------------

Plotting Calibration Curves
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   from sklearn.calibration import calibration_curve
   
   def plot_calibration_curve(y_true, y_prob_list, names, n_bins=10):
       """Plot calibration curves for multiple methods."""
       fig, ax = plt.subplots(figsize=(10, 8))
       
       # Perfect calibration line
       ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
       
       # Plot each method
       for y_prob, name in zip(y_prob_list, names):
           fraction_pos, mean_pred = calibration_curve(
               y_true, y_prob, n_bins=n_bins
           )
           ax.plot(mean_pred, fraction_pos, 'o-', label=name)
       
       ax.set_xlabel('Mean Predicted Probability')
       ax.set_ylabel('Fraction of Positives')
       ax.set_title('Calibration Plot (Reliability Diagram)')
       ax.legend()
       ax.grid(True, alpha=0.3)
       return fig, ax
   
   # Plot comparison
   y_prob_list = [y_pred_uncal, y_pred_cal]
   names = ['Uncalibrated', 'Nearly Isotonic']
   
   fig, ax = plot_calibration_curve(y_test, y_prob_list, names)
   plt.show()

Distribution Plots
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def plot_prediction_distributions(y_prob_list, names):
       """Plot prediction distributions."""
       fig, axes = plt.subplots(1, len(y_prob_list), figsize=(15, 5))
       if len(y_prob_list) == 1:
           axes = [axes]
           
       for i, (y_prob, name) in enumerate(zip(y_prob_list, names)):
           axes[i].hist(y_prob, bins=50, alpha=0.7, density=True)
           axes[i].set_title(f'{name}\\nUnique values: {len(np.unique(y_prob))}')
           axes[i].set_xlabel('Predicted Probability')
           axes[i].set_ylabel('Density')
           axes[i].grid(True, alpha=0.3)
       
       plt.tight_layout()
       return fig, axes
   
   # Plot distributions
   fig, axes = plot_prediction_distributions(y_prob_list, names)
   plt.show()

Cross-Validation for Calibration
---------------------------------

.. code-block:: python

   from sklearn.model_selection import cross_val_predict
   from sklearn.base import clone
   
   def cross_validated_calibration(model, calibrator, X, y, cv=5):
       """Perform cross-validated calibration."""
       # Get cross-validated predictions
       y_pred_cv = cross_val_predict(
           model, X, y, cv=cv, method='predict_proba'
       )[:, 1]
       
       # Split for calibration training and testing
       X_cal_train, X_cal_test, y_cal_train, y_cal_test = train_test_split(
           y_pred_cv.reshape(-1, 1), y, test_size=0.5, random_state=42
       )
       
       # Fit calibrator
       cal_clone = clone(calibrator)
       cal_clone.fit(X_cal_train.ravel(), y_cal_train)
       
       # Get calibrated predictions
       y_cal_pred = cal_clone.transform(X_cal_test.ravel())
       
       return y_cal_test, X_cal_test.ravel(), y_cal_pred
   
   # Perform cross-validated calibration
   y_true_cv, y_pred_uncal_cv, y_pred_cal_cv = cross_validated_calibration(
       model, NearlyIsotonicCalibrator(lam=1.0), X, y
   )
   
   print("Cross-validated results:")
   print(f"MCE uncalibrated: {mean_calibration_error(y_true_cv, y_pred_uncal_cv):.4f}")
   print(f"MCE calibrated: {mean_calibration_error(y_true_cv, y_pred_cal_cv):.4f}")

Common Pitfalls and Solutions
-----------------------------

Avoiding Overfitting in Calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # DON'T: Use the same data for training and calibration
   model.fit(X_train, y_train)
   y_pred = model.predict_proba(X_train)[:, 1]  # Same data!
   calibrator.fit(y_pred, y_train)  # This will overfit
   
   # DO: Use separate data or cross-validation
   model.fit(X_train, y_train)
   y_pred = model.predict_proba(X_test)[:, 1]  # Different data
   calibrator.fit(y_pred, y_test)  # Better approach

Handling Edge Cases
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Check for problematic predictions
   def validate_predictions(y_pred):
       """Validate prediction array."""
       if np.any(y_pred < 0) or np.any(y_pred > 1):
           print("Warning: Predictions outside [0,1] range")
       
       if len(np.unique(y_pred)) < 10:
           print("Warning: Very few unique prediction values")
       
       if np.any(np.isnan(y_pred)):
           print("Warning: NaN values in predictions")
   
   validate_predictions(y_pred_uncal)
   
   # Handle constant predictions
   if len(np.unique(y_pred_uncal)) == 1:
       print("Constant predictions detected - calibration may not be meaningful")
   else:
       calibrator.fit(y_pred_uncal, y_test)