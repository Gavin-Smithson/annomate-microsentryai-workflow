import numpy as np
import pytest

from core.states.inference_state import InferenceState


@pytest.fixture
def state():
    return InferenceState()


@pytest.fixture
def score_map():
    arr = np.array([[0.1, 0.5], [0.9, 0.3]], dtype=np.float32)
    return arr


class TestInferenceStateDefaults:
    def test_score_maps_empty_on_init(self, state):
        assert state.score_maps == {}

    def test_inference_cache_empty_on_init(self, state):
        assert state.inference_cache == {}

    def test_is_processed_false_for_unknown(self, state):
        assert state.is_processed("missing.jpg") is False

    def test_get_score_map_none_for_unknown(self, state):
        assert state.get_score_map("missing.jpg") is None


class TestInferenceStateMutations:
    def test_set_score_map_stores_array(self, state, score_map):
        state.set_score_map("img.jpg", score_map)
        stored = state.get_score_map("img.jpg")
        assert stored is not None
        np.testing.assert_array_equal(stored, score_map)

    def test_is_processed_true_after_set(self, state, score_map):
        state.set_score_map("img.jpg", score_map)
        assert state.is_processed("img.jpg") is True

    def test_set_score_map_caches_peak_value(self, state, score_map):
        state.set_score_map("img.jpg", score_map)
        assert state.inference_cache["img.jpg"] == pytest.approx(float(score_map.max()))

    def test_all_zeros_map_caches_zero(self, state):
        state.set_score_map("blank.jpg", np.zeros((4, 4), dtype=np.float32))
        assert state.inference_cache["blank.jpg"] == pytest.approx(0.0)

    def test_multiple_images_stored_independently(self, state):
        a = np.full((2, 2), 0.2, dtype=np.float32)
        b = np.full((2, 2), 0.8, dtype=np.float32)
        state.set_score_map("a.jpg", a)
        state.set_score_map("b.jpg", b)
        assert state.is_processed("a.jpg")
        assert state.is_processed("b.jpg")
        assert state.inference_cache["a.jpg"] == pytest.approx(0.2)
        assert state.inference_cache["b.jpg"] == pytest.approx(0.8)

    def test_overwrite_updates_peak_cache(self, state):
        state.set_score_map("img.jpg", np.full((2, 2), 0.3, dtype=np.float32))
        state.set_score_map("img.jpg", np.full((2, 2), 0.9, dtype=np.float32))
        assert state.inference_cache["img.jpg"] == pytest.approx(0.9)


class TestInferenceStateClear:
    def test_clear_empties_score_maps(self, state, score_map):
        state.set_score_map("img.jpg", score_map)
        state.clear()
        assert state.score_maps == {}

    def test_clear_empties_inference_cache(self, state, score_map):
        state.set_score_map("img.jpg", score_map)
        state.clear()
        assert state.inference_cache == {}

    def test_is_processed_false_after_clear(self, state, score_map):
        state.set_score_map("img.jpg", score_map)
        state.clear()
        assert state.is_processed("img.jpg") is False

    def test_can_store_after_clear(self, state, score_map):
        state.set_score_map("img.jpg", score_map)
        state.clear()
        state.set_score_map("new.jpg", score_map)
        assert state.is_processed("new.jpg")
