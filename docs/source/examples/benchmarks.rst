Performance Benchmarks
====================

This section provides performance comparisons and benchmarks for different calibration methods.

Interactive Performance Notebook
---------------------------------

The most comprehensive benchmarks are available in our interactive Jupyter notebook:

- **Location**: :doc:`../notebooks/04_performance_comparison` in the documentation
- **Content**: Systematic comparison across all calibration methods
- **Features**: Visual comparisons, quantitative metrics, computational efficiency analysis

Accessing the Performance Notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone repository
   git clone https://github.com/finite-sample/calibre.git
   cd calibre
   
   # Install dependencies
   pip install -e ".[dev]"
   
   # Start Jupyter
   jupyter notebook docs/source/notebooks/04_performance_comparison.ipynb

Or view it online in the documentation: :doc:`../notebooks/04_performance_comparison`

Method Comparison Summary
-------------------------

Based on extensive benchmarking across different datasets and scenarios:

Performance Summary Table
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Calibration Method Performance
   :header-rows: 1
   :widths: 25 15 15 15 15 15

   * - Method
     - Calibration Error
     - Granularity Preservation
     - Computational Speed
     - Robustness
     - Use Case
   * - Nearly Isotonic (strict)
     - ★★★★★
     - ★★☆☆☆
     - ★★★☆☆
     - ★★★★☆
     - High-stakes decisions
   * - Nearly Isotonic (relaxed)
     - ★★★★☆
     - ★★★★☆
     - ★★★☆☆
     - ★★★★☆
     - Balanced approach
   * - I-Spline
     - ★★★★☆
     - ★★★☆☆
     - ★★☆☆☆
     - ★★★☆☆
     - Smooth calibration
   * - Relaxed PAVA
     - ★★★☆☆
     - ★★★★★
     - ★★★★★
     - ★★★★★
     - Large datasets
   * - Regularized Isotonic
     - ★★★☆☆
     - ★★★☆☆
     - ★★★★☆
     - ★★★☆☆
     - Smooth results needed
   * - Smoothed Isotonic
     - ★★★☆☆
     - ★★★☆☆
     - ★★★★☆
     - ★★★★☆
     - Visualization

Detailed Performance Analysis
----------------------------

Calibration Error Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.datasets import make_classification
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   
   from calibre import (
       NearlyIsotonicCalibrator,
       SplineCalibrator,
       RelaxedPAVACalibrator,
       RegularizedIsotonicRegression,
       SmoothedIsotonicRegression,
       mean_calibration_error,
       expected_calibration_error
   )
   
   def comprehensive_benchmark(n_datasets=10, n_samples=2000):
       """Run comprehensive benchmark across multiple datasets."""
       
       calibrators = {
           'Nearly Isotonic (λ=10)': NearlyIsotonicCalibrator(lam=10.0, method='path'),
           'Nearly Isotonic (λ=1)': NearlyIsotonicCalibrator(lam=1.0, method='path'),
           'Nearly Isotonic (λ=0.1)': NearlyIsotonicCalibrator(lam=0.1, method='path'),
           'I-Spline': SplineCalibrator(n_splines=10, degree=3, cv=3),
           'Relaxed PAVA': RelaxedPAVACalibrator(percentile=10, adaptive=True),
           'Regularized Isotonic': RegularizedIsotonicRegression(alpha=0.1),
           'Smoothed Isotonic': SmoothedIsotonicRegression(window_length=7, poly_order=3)
       }
       
       results = {name: {'mce': [], 'ece': [], 'time': []} for name in calibrators.keys()}
       
       for dataset_idx in range(n_datasets):
           print(f"\\rProcessing dataset {dataset_idx + 1}/{n_datasets}", end='')
           
           # Generate dataset with varying characteristics
           X, y = make_classification(
               n_samples=n_samples,
               n_features=20,
               n_informative=15,
               n_redundant=2,
               random_state=dataset_idx * 42
           )
           
           X_train, X_test, y_train, y_test = train_test_split(
               X, y, test_size=0.5, random_state=dataset_idx
           )
           
           # Train base model
           model = RandomForestClassifier(n_estimators=100, random_state=dataset_idx)
           model.fit(X_train, y_train)
           y_pred = model.predict_proba(X_test)[:, 1]
           
           # Test each calibrator
           for name, calibrator in calibrators.items():
               try:
                   import time
                   start_time = time.time()
                   
                   # Fit and transform
                   calibrator.fit(y_pred, y_test)
                   y_cal = calibrator.transform(y_pred)
                   
                   end_time = time.time()
                   
                   # Calculate metrics
                   mce = mean_calibration_error(y_test, y_cal)
                   ece = expected_calibration_error(y_test, y_cal, n_bins=10)
                   
                   results[name]['mce'].append(mce)
                   results[name]['ece'].append(ece)
                   results[name]['time'].append(end_time - start_time)
                   
               except Exception as e:
                   print(f"\\nError with {name}: {e}")
                   results[name]['mce'].append(np.nan)
                   results[name]['ece'].append(np.nan)
                   results[name]['time'].append(np.nan)
       
       print()  # New line after progress
       return results
   
   # Run benchmark
   benchmark_results = comprehensive_benchmark(n_datasets=5, n_samples=1000)
   
   # Display results
   print("\\nBenchmark Results (Mean ± Std):")
   print(f"{'Method':<25} {'MCE':<15} {'ECE':<15} {'Time (ms)':<15}")
   print("-" * 75)
   
   for name, metrics in benchmark_results.items():
       mce_mean = np.nanmean(metrics['mce'])
       mce_std = np.nanstd(metrics['mce'])
       ece_mean = np.nanmean(metrics['ece'])
       ece_std = np.nanstd(metrics['ece'])
       time_mean = np.nanmean(metrics['time']) * 1000  # Convert to ms
       time_std = np.nanstd(metrics['time']) * 1000
       
       print(f"{name:<25} {mce_mean:.3f}±{mce_std:.3f}    "
             f"{ece_mean:.3f}±{ece_std:.3f}    {time_mean:.1f}±{time_std:.1f}")

Scalability Analysis
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def scalability_benchmark():
       """Test performance across different dataset sizes."""
       
       dataset_sizes = [500, 1000, 2000, 5000, 10000]
       methods = {
           'Nearly Isotonic': NearlyIsotonicCalibrator(lam=1.0, method='path'),
           'Relaxed PAVA': RelaxedPAVACalibrator(percentile=10),
           'Regularized Isotonic': RegularizedIsotonicRegression(alpha=0.1)
       }
       
       timing_results = {method: [] for method in methods.keys()}
       
       for n_samples in dataset_sizes:
           print(f"Testing with {n_samples} samples...")
           
           # Generate data
           X, y = make_classification(n_samples=n_samples, n_features=20, random_state=42)
           X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
           
           # Train model
           model = RandomForestClassifier(n_estimators=50, random_state=42)
           model.fit(X_train, y_train)
           y_pred = model.predict_proba(X_test)[:, 1]
           
           for method_name, calibrator in methods.items():
               import time
               
               # Time the calibration process
               start_time = time.time()
               calibrator.fit(y_pred, y_test)
               y_cal = calibrator.transform(y_pred)
               end_time = time.time()
               
               timing_results[method_name].append(end_time - start_time)
       
       # Plot results
       plt.figure(figsize=(10, 6))
       for method_name, times in timing_results.items():
           plt.plot(dataset_sizes, times, 'o-', label=method_name, linewidth=2)
       
       plt.xlabel('Dataset Size')
       plt.ylabel('Time (seconds)')
       plt.title('Calibration Method Scalability')
       plt.legend()
       plt.grid(True, alpha=0.3)
       plt.yscale('log')
       plt.show()
       
       return timing_results
   
   # Run scalability test
   scalability_results = scalability_benchmark()

Dataset-Specific Performance
---------------------------

Performance on Different Data Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def dataset_specific_benchmark():
       """Test performance on different types of datasets."""
       
       datasets = {
           'balanced': lambda: make_classification(
               n_samples=2000, n_features=20, weights=[0.5, 0.5], random_state=42
           ),
           'imbalanced': lambda: make_classification(
               n_samples=2000, n_features=20, weights=[0.9, 0.1], random_state=42
           ),
           'high_dim': lambda: make_classification(
               n_samples=2000, n_features=100, n_informative=20, random_state=42
           ),
           'low_info': lambda: make_classification(
               n_samples=2000, n_features=20, n_informative=5, n_redundant=10, random_state=42
           )
       }
       
       calibrators = {
           'Nearly Isotonic': NearlyIsotonicCalibrator(lam=1.0),
           'Relaxed PAVA': RelaxedPAVACalibrator(percentile=10),
           'I-Spline': SplineCalibrator(n_splines=8, cv=3)
       }
       
       results = {}
       
       for dataset_name, dataset_func in datasets.items():
           print(f"\\nTesting on {dataset_name} dataset:")
           
           X, y = dataset_func()
           X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
           
           # Train model
           model = RandomForestClassifier(n_estimators=100, random_state=42)
           model.fit(X_train, y_train)
           y_pred = model.predict_proba(X_test)[:, 1]
           
           dataset_results = {}
           
           for cal_name, calibrator in calibrators.items():
               try:
                   calibrator.fit(y_pred, y_test)
                   y_cal = calibrator.transform(y_pred)
                   
                   mce = mean_calibration_error(y_test, y_cal)
                   ece = expected_calibration_error(y_test, y_cal)
                   
                   dataset_results[cal_name] = {'mce': mce, 'ece': ece}
                   print(f"  {cal_name}: MCE={mce:.4f}, ECE={ece:.4f}")
                   
               except Exception as e:
                   print(f"  {cal_name}: Failed - {e}")
                   dataset_results[cal_name] = {'mce': np.nan, 'ece': np.nan}
           
           results[dataset_name] = dataset_results
       
       return results
   
   # Run dataset-specific benchmark
   dataset_results = dataset_specific_benchmark()

Robustness Analysis
------------------

Noise Sensitivity
~~~~~~~~~~~~~~~~

.. code-block:: python

   def noise_sensitivity_test():
       """Test calibrator robustness to different noise levels."""
       
       noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]
       calibrators = {
           'Nearly Isotonic': NearlyIsotonicCalibrator(lam=1.0),
           'Relaxed PAVA': RelaxedPAVACalibrator(percentile=15),  # Slightly higher for noise
           'Regularized Isotonic': RegularizedIsotonicRegression(alpha=0.5)
       }
       
       results = {name: [] for name in calibrators.keys()}
       
       for noise_level in noise_levels:
           print(f"Testing noise level: {noise_level}")
           
           # Generate clean data
           X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
           X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
           
           # Train model
           model = RandomForestClassifier(n_estimators=100, random_state=42)
           model.fit(X_train, y_train)
           y_pred_clean = model.predict_proba(X_test)[:, 1]
           
           # Add noise to predictions
           noise = np.random.normal(0, noise_level, len(y_pred_clean))
           y_pred_noisy = np.clip(y_pred_clean + noise, 0, 1)
           
           for name, calibrator in calibrators.items():
               try:
                   calibrator.fit(y_pred_noisy, y_test)
                   y_cal = calibrator.transform(y_pred_noisy)
                   mce = mean_calibration_error(y_test, y_cal)
                   results[name].append(mce)
               except:
                   results[name].append(np.nan)
       
       # Plot results
       plt.figure(figsize=(10, 6))
       for name, mce_values in results.items():
           plt.plot(noise_levels, mce_values, 'o-', label=name, linewidth=2)
       
       plt.xlabel('Noise Level')
       plt.ylabel('Mean Calibration Error')
       plt.title('Robustness to Prediction Noise')
       plt.legend()
       plt.grid(True, alpha=0.3)
       plt.show()
       
       return results
   
   # Run noise sensitivity test
   noise_results = noise_sensitivity_test()

Memory Usage Analysis
--------------------

.. code-block:: python

   import psutil
   import os
   
   def memory_usage_benchmark():
       """Analyze memory usage of different calibrators."""
       
       def get_memory_usage():
           process = psutil.Process(os.getpid())
           return process.memory_info().rss / 1024 / 1024  # MB
       
       # Generate large dataset
       X, y = make_classification(n_samples=50000, n_features=20, random_state=42)
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
       
       model = RandomForestClassifier(n_estimators=100, random_state=42)
       model.fit(X_train, y_train)
       y_pred = model.predict_proba(X_test)[:, 1]
       
       calibrators = {
           'Nearly Isotonic (CVX)': NearlyIsotonicCalibrator(lam=1.0, method='cvx'),
           'Nearly Isotonic (Path)': NearlyIsotonicCalibrator(lam=1.0, method='path'),
           'Relaxed PAVA': RelaxedPAVACalibrator(percentile=10),
           'Regularized Isotonic': RegularizedIsotonicRegression(alpha=0.1)
       }
       
       memory_results = {}
       
       for name, calibrator in calibrators.items():
           print(f"Testing memory usage for {name}...")
           
           # Measure baseline memory
           baseline_memory = get_memory_usage()
           
           try:
               # Fit calibrator
               calibrator.fit(y_pred, y_test)
               
               # Measure peak memory
               peak_memory = get_memory_usage()
               
               # Transform data
               y_cal = calibrator.transform(y_pred)
               
               # Measure final memory
               final_memory = get_memory_usage()
               
               memory_results[name] = {
                   'peak_usage': peak_memory - baseline_memory,
                   'final_usage': final_memory - baseline_memory
               }
               
           except Exception as e:
               print(f"Failed: {e}")
               memory_results[name] = {'peak_usage': np.nan, 'final_usage': np.nan}
       
       # Display results
       print("\\nMemory Usage Results:")
       print(f"{'Method':<25} {'Peak (MB)':<12} {'Final (MB)':<12}")
       print("-" * 50)
       
       for name, usage in memory_results.items():
           print(f"{name:<25} {usage['peak_usage']:<12.1f} {usage['final_usage']:<12.1f}")
       
       return memory_results
   
   # Run memory benchmark
   memory_results = memory_usage_benchmark()

Benchmark Reproduction
---------------------

To reproduce these benchmarks:

1. **Install Calibre with development dependencies**:

   .. code-block:: bash

      pip install -e ".[dev]"

2. **Run the interactive performance comparison notebook**:

   .. code-block:: bash

      jupyter notebook docs/source/notebooks/04_performance_comparison.ipynb

3. **Execute individual benchmark functions** from this documentation

4. **Customize benchmarks** for your specific datasets and use cases

Our interactive notebooks provide comprehensive visualizations, systematic comparisons, and detailed analysis. See :doc:`../notebooks/04_performance_comparison` for the most current benchmarks, and :doc:`index` for all available examples.