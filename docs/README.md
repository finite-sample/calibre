# Calibre Documentation

This directory contains the Sphinx documentation for Calibre.

## Building Documentation Locally

### Option 1: Using the package installation (recommended)

```bash
# Install Calibre with documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html

# View documentation
open build/html/index.html  # macOS
# or
xdg-open build/html/index.html  # Linux
# or navigate to docs/build/html/index.html in your browser
```

### Option 2: Using requirements file

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build documentation
cd docs
make html
```

### Option 3: Using make commands

```bash
cd docs

# Clean previous builds
make clean

# Build HTML documentation
make html

# Build PDF documentation (requires LaTeX)
make latexpdf

# Check for broken links
make linkcheck

# Show all available targets
make help
```

## Documentation Structure

```
docs/
├── source/
│   ├── conf.py              # Sphinx configuration
│   ├── index.rst            # Main documentation page
│   ├── installation.rst     # Installation guide
│   ├── quickstart.rst       # Quick start guide
│   ├── contributing.rst     # Contributing guide
│   ├── api/                 # API documentation
│   │   ├── index.rst
│   │   ├── calibrators.rst
│   │   ├── metrics.rst
│   │   └── utils.rst
│   ├── examples/            # Examples and tutorials
│   │   ├── index.rst
│   │   ├── basic_usage.rst
│   │   ├── advanced_usage.rst
│   │   └── benchmarks.rst
│   └── _static/             # Static assets (CSS, images)
│       └── custom.css
├── build/                   # Generated documentation (gitignored)
├── Makefile                 # Unix build commands
├── make.bat                 # Windows build commands
├── requirements.txt         # Documentation dependencies
└── README.md               # This file
```

## Auto-Generated Documentation

The API documentation is automatically generated from docstrings in the source code using Sphinx's autodoc extension. To update the API documentation:

1. Make sure your code has proper docstrings (NumPy/Google style)
2. Rebuild the documentation with `make html`
3. The API pages will be updated automatically

## GitHub Pages Deployment

Documentation is automatically deployed to GitHub Pages when changes are pushed to the main branch. The deployment workflow is defined in `.github/workflows/docs.yml`.

**Live documentation**: https://finite-sample.github.io/calibre/

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install the package in development mode first:
   ```bash
   pip install -e .
   ```

2. **Sphinx build warnings**: These are usually harmless but can be fixed by:
   - Adding missing docstrings
   - Fixing import paths
   - Checking reStructuredText syntax

3. **Missing notebook output**: Notebooks are not executed during build by default. To include outputs, change `nbsphinx_execute = 'never'` to `nbsphinx_execute = 'always'` in `conf.py`.

### Getting Help

- Check the [Sphinx documentation](https://www.sphinx-doc.org/)
- Review existing documentation files for examples
- Open an issue on GitHub if you encounter problems

## Contributing to Documentation

1. Follow the existing structure and style
2. Use reStructuredText (.rst) format for main pages
3. Include code examples that actually work
4. Test your changes locally before submitting
5. Update the table of contents if adding new pages

For more details, see the [Contributing Guide](source/contributing.rst).