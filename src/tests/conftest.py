import pytest
from core.states.dataset_state import DatasetState


@pytest.fixture
def state():
    return DatasetState()
