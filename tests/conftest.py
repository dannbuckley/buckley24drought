from typing import List

import pytest

SUITES = ("slow", "intg")


def pytest_addoption(parser: pytest.Parser):
    """A pytest hook function that allows us to add additional
    command line arguments and manage which tests are run.

    Parameters
    ----------
    parser : pytest.Parser
    """
    for suite in SUITES:
        parser.addoption(
            f"--{suite}",
            action="store_true",
            default=False,
            help=f"run {suite} tests",
        )

    parser.addoption(
        "--all",
        action="store_true",
        default=False,
        help="run all tests",
    )


def pytest_configure(config: pytest.Config):
    """Mark tests with the suite name.

    Parameters
    ----------
    config : pytest.Config
    """
    for suite in SUITES:
        config.addinivalue_line("markers", f"{suite}: mark test as {suite}")


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]):
    """Skip tests that are not explicitly under the given suite.

    Parameters
    ----------
    config : pytest.Config
    items : list of pytest.Item
    """
    skips = {suite: pytest.mark.skip(reason=f"--{suite} not set") for suite in SUITES}
    run = set(name for name in SUITES if config.getoption(f"--{name}"))

    if config.getoption("--all"):
        run = set(SUITES)

    for item in items:
        if not set(item.keywords) & run:
            for suitename in set(SUITES) & set(item.keywords):
                item.add_marker(skips[suitename])
