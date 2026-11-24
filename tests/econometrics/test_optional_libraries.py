"""Econometrics without the optional statsmodels and linearmodels libraries."""

import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]

# Run in a fresh interpreter in which statsmodels and linearmodels cannot be imported, as the
# test session itself has imported them already.
CODE = """
import sys

for library in ("statsmodels", "linearmodels"):
    sys.modules[library] = None

import pandas as pd

from financetoolkit import Econometrics

prices = pd.read_pickle("tests/datasets/historical_dataset.pickle")["Adj Close"]
econometrics = Econometrics(data=prices[["AAPL", "MSFT"]])

ols = econometrics.get_ols()
print("ols", ols.shape[0], int(ols["Coefficient"].notna().sum()))

try:
    econometrics.get_unit_root_test()
    print("adf", "no error")
except ImportError as error:
    print("adf", error)
"""


def test_econometrics_works_without_statsmodels_and_linearmodels():
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(REPOSITORY), os.environ.get("PYTHONPATH")])
        ),
    }
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", CODE],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPOSITORY,
        env=environment,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr

    lines = dict(line.split(" ", 1) for line in result.stdout.strip().splitlines())

    # Ordinary least squares is our own code: it returns coefficients without the libraries.
    rows, coefficients = map(int, lines["ols"].split())
    assert rows > 0
    assert coefficients > 0

    # The ADF test is statsmodels': it says which library it needs and how to install it.
    assert lines["adf"].startswith(
        "The Augmented Dickey-Fuller test needs statsmodels."
    )
    assert "pip install financetoolkit[econometrics]" in lines["adf"]
