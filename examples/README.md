# Examples

This directory contains example notebooks demonstrating the calibre package functionality.

## Contents

- `benchmark.ipynb` - Performance comparison of different calibration methods across various scenarios
- `validation/calibration_validation.ipynb` - Visual validation of calibration methods with reliability diagrams

## Usage

Install the required dependencies:
```bash
pip install -e ".[dev]"
pip install jupyter matplotlib
```

Then run the notebooks:
```bash
jupyter notebook examples/
```

These examples are provided for demonstration and educational purposes. For automated testing and validation, see the `tests/` directory.