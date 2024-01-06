# buckley24drought
Data and code for 'Investigating the drought sensitivity of intermontane-basin groundwater systems in western Montana using standardized drought indices' (Buckley, 2024)

## Development
All Python code in `src/buckley24drought/` is linted using [Pylint](https://pypi.org/project/pylint/) on every push and PR.

All unit tests should go in the `tests/` directory. All tests are run using [pytest](https://docs.pytest.org/en/7.4.x/) on every push and PR. Install the package in "editable" mode to run the tests locally:

```
pip install -e .
```

Docstrings should follow the [numpydoc style guide](https://numpydoc.readthedocs.io/en/latest/format.html).