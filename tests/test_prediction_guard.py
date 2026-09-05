from app import main


def test_prediction_guard_accepts_clear_prediction():
    result = main.build_prediction_payload(
        class_id=67,
        flower_name="sunflower",
        confidence=0.82,
        margin=0.35,
    )

    assert result["recognized"] is True
    assert result["flower_name"] == "SUNFLOWER"
    assert result["message"] == "Flower recognized."


def test_prediction_guard_rejects_uncertain_prediction():
    result = main.build_prediction_payload(
        class_id=12,
        flower_name="rose",
        confidence=0.41,
        margin=0.04,
    )

    assert result["recognized"] is False
    assert result["flower_name"] == "UNKNOWN"
    assert "not a flower" in result["message"].lower()
