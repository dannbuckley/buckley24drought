import pytest

SUITES = ("slow", "intg")

def pytest_addoption(parser):
    """
    A pytest hook function that allows us to add additional command line arguments
    and manage which tests are run.
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

def pytest_configure(config):
    """
    Mark tests with the suite name.
    """
    for suite in SUITES:
        config.addinivalue_line(
            "markers", f"{suite}: mark test as {suite}"
        )

def pytest_collection_modifyitems(config, items):
    """
    Skip tests that are not explicitly under the given suite.
    """
    skips = {suite: pytest.mark.skip(reason=f"--{suite} not set") for suite in SUITES}
    run = set(name for name in SUITES if config.getoption(f"--{name}"))

    if config.getoption("--all"):
        run = set(SUITES)

    for item in items:
        if not set(item.keywords) & run:
            for suitename in set(SUITES) & set(item.keywords):
                item.add_marker(skips[suitename])
