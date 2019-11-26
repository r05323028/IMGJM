import pytest
from IMGJM import IMGJM


class TestModel:
    @pytest.fixture()
    def fake_data(self):
        char_ids = None
        word_ids = None
        sequence_length = None
        y_target = None
        y_sentiment = None
        return char_ids, word_ids, sequence_length, y_target, y_sentiment

    def test_model(self, fake_data):
        return NotImplemented