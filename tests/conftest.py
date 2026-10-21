import pytest

from nflmodel.config import load_config


@pytest.fixture
def cfg() -> dict:
    return load_config()
