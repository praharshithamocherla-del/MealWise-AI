
import pytest
from src.recommendations import make_recommendation

def test_recommendation_rounds_up_with_margin():
    result = make_recommendation(100.2, 10, 90)
    assert result["predicted_attendance"] == 100
    assert result["recommended_prepare"] == 110
    assert result["estimated_shortage"] == 10

def test_invalid_margin_is_rejected():
    with pytest.raises(ValueError):
        make_recommendation(100, 45, 100)
