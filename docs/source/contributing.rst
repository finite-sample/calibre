Contributing to Calibre
======================

We welcome contributions to Calibre! This document provides guidelines for contributing to the project.

Types of Contributions
----------------------

We welcome several types of contributions:

- **Bug reports**: Help us identify and fix issues
- **Feature requests**: Suggest new calibration methods or improvements
- **Code contributions**: Implement new features or fix bugs
- **Documentation**: Improve or expand the documentation
- **Examples**: Add usage examples or tutorials

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/yourusername/calibre.git
      cd calibre

3. Create a development environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      pip install -e ".[dev]"

4. Create a feature branch:

   .. code-block:: bash

      git checkout -b feature-name

Development Workflow
--------------------

Code Style
~~~~~~~~~~

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting  
- **flake8**: Linting
- **mypy**: Type checking (optional but recommended)

Run these before submitting:

.. code-block:: bash

   # Format code
   black calibre/ tests/
   
   # Sort imports
   isort calibre/ tests/
   
   # Check linting
   flake8 calibre/ tests/

Testing
~~~~~~~

All contributions should include tests. We use pytest for testing:

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run with coverage
   pytest --cov=calibre
   
   # Run specific test file
   pytest tests/test_calibrators_unit.py

**Note**: Some tests may be skipped when calibrators reach their mathematical limits. This is expected behavior - typically 6-8 tests are skipped out of ~140 total tests.

Documentation
~~~~~~~~~~~~~

If you're adding new features, please include documentation:

1. Add docstrings to new functions/classes using NumPy style
2. Update relevant .rst files in ``docs/source/``
3. Build docs locally to check formatting:

   .. code-block:: bash

      cd docs
      make html
      # View docs/build/html/index.html in browser

Code Guidelines
---------------

API Design
~~~~~~~~~~

- Follow scikit-learn conventions (fit/transform pattern)
- Inherit from ``BaseCalibrator`` for new calibration methods
- Use type hints throughout your code
- Include comprehensive docstrings

Example for a new calibrator:

.. code-block:: python

   class MyCalibrator(BaseCalibrator):
       """Short description of the calibrator.
       
       Longer description explaining the method, when to use it,
       and any important implementation details.
       
       Parameters
       ----------
       param1 : float, default=1.0
           Description of parameter.
       param2 : str, default='auto'
           Description of parameter.
           
       Attributes
       ----------
       X_ : ndarray of shape (n_samples,)
           The training input samples.
       y_ : ndarray of shape (n_samples,)
           The target values.
           
       Examples
       --------
       >>> from calibre import MyCalibrator
       >>> cal = MyCalibrator(param1=2.0)
       >>> cal.fit(X, y)
       >>> y_calibrated = cal.transform(X)
       """
       
       def __init__(self, param1: float = 1.0, param2: str = 'auto'):
           self.param1 = param1
           self.param2 = param2
           
       def fit(self, X: np.ndarray, y: np.ndarray) -> "MyCalibrator":
           # Implementation
           return self
           
       def transform(self, X: np.ndarray) -> np.ndarray:
           # Implementation
           return calibrated_predictions

Testing Guidelines
~~~~~~~~~~~~~~~~~~

- Write tests for all new functionality
- Include edge cases and error conditions
- Test with different data types and sizes
- Use pytest fixtures for common test data

Example test structure:

.. code-block:: python

   def test_my_calibrator_basic():
       """Test basic functionality."""
       cal = MyCalibrator()
       X, y = generate_test_data()
       cal.fit(X, y)
       y_cal = cal.transform(X)
       
       assert len(y_cal) == len(X)
       assert np.all(y_cal >= 0) and np.all(y_cal <= 1)
       
   def test_my_calibrator_edge_cases():
       """Test edge cases."""
       cal = MyCalibrator()
       
       # Test with constant predictions
       X = np.array([0.5] * 100)
       y = np.array([1] * 100)
       cal.fit(X, y)
       y_cal = cal.transform(X)
       
       # Should handle gracefully
       assert not np.any(np.isnan(y_cal))

Pull Request Process
--------------------

1. **Before submitting**:
   
   - Run all tests locally and ensure they pass
   - Run code quality checks (black, isort, flake8)
   - Update documentation if needed
   - Add tests for new functionality

2. **Pull request description**:
   
   - Clearly describe what the PR does
   - Reference any related issues
   - Include examples of usage if applicable
   - List any breaking changes

3. **Review process**:
   
   - Maintainers will review your PR
   - Address any feedback promptly
   - CI tests must pass before merging

Example PR template:

.. code-block::

   ## Description
   Brief description of changes
   
   ## Type of change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement
   
   ## Testing
   - [ ] Added tests for new functionality
   - [ ] All existing tests pass
   - [ ] Manual testing completed
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] No breaking changes (or clearly documented)

Reporting Issues
----------------

When reporting bugs, please include:

1. **Environment information**:
   - Python version
   - Calibre version
   - Operating system
   - Dependency versions (numpy, scipy, etc.)

2. **Minimal reproduction example**:
   - Code that reproduces the issue
   - Sample data if relevant
   - Expected vs actual behavior

3. **Error messages**:
   - Full traceback
   - Any relevant warnings

Issue template:

.. code-block::

   **Bug Description**
   Clear description of the bug
   
   **To Reproduce**
   Steps to reproduce the behavior:
   1. ...
   2. ...
   
   **Expected Behavior**
   What you expected to happen
   
   **Environment**
   - OS: [e.g. macOS 12.0]
   - Python: [e.g. 3.10.2]
   - Calibre: [e.g. 0.3.0]
   
   **Additional Context**
   Any other relevant information

Release Process
---------------

For maintainers, here's the release process:

1. Update version in ``pyproject.toml``
2. Update ``CHANGELOG.md``
3. Create release tag
4. GitHub Actions will automatically build and publish to PyPI

Getting Help
------------

If you need help with contributing:

- Open a discussion on GitHub
- Check existing issues and PRs
- Ask questions in your PR if you're unsure about implementation

Thank you for contributing to Calibre!