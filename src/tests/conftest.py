import pytest
from core.dataset_state import DatasetState


@pytest.fixture
def state():
    return DatasetState()
