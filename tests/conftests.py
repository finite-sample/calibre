"""
Configuration for pytest to properly collect only our tests.
"""

import sys
from pathlib import Path

# Add parent directory to path to ensure we can import the package
parent_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(parent_dir))


def pytest_collection_modifyitems(items, config):
    """Filter test items to include only our package tests."""
    selected_items = []
    for item in items:
        # Only include test files from our package
        if str(parent_dir) in str(item.fspath):
            selected_items.append(item)
    items[:] = selected_items


def pytest_ignore_collect(path, config):
    """Prevent collecting tests from system directories."""
    path_str = str(path)
    # Explicitly ignore problematic directories
    ignore_patterns = [
        "/Applications/",
        "/opt/homebrew/",
        "/Library/",
        "/System/",
        "site-packages/",
        "idlelib",
        "tkinter",
        "TeX Live",
    ]
    return any(pattern in path_str for pattern in ignore_patterns)
