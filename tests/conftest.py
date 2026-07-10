import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: long-running integration tests against bunny/data")


@pytest.fixture
def bunny_data_dir() -> str:
    return "bunny/data"
