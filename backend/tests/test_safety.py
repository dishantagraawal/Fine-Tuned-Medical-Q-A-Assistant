from app.services.safety import analyze_user_message, emergency_response, harmful_response


def test_emergency_detection():
    flags = analyze_user_message("I have severe chest pain and can't breathe")
    assert flags["is_emergency"] is True
    assert flags["block_generation"] is True
    assert "emergency" in emergency_response().lower()


def test_normal_question():
    flags = analyze_user_message("What is hypertension?")
    assert flags["is_emergency"] is False
    assert flags["block_generation"] is False


def test_harmful_block():
    flags = analyze_user_message("how to commit suicide with pills")
    assert flags["is_harmful"] is True
    assert "cannot help" in harmful_response().lower()
