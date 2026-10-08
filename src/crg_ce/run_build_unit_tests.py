"""Run unit build tests using pytest.

This script runs all tests marked with @pytest.mark.unit_build.
"""

import sys

import pytest

if __name__ == "__main__":
    # Run pytest with the unit_build marker
    # -v for verbose output
    # -m unit_build to only run tests marked with @pytest.mark.unit_build
    exit_code = pytest.main(
        [
            "-v",
            "-m",
            "unit_build",
            "--tb=short",
        ]
    )
    sys.exit(exit_code)
