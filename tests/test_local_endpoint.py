import pytest
import requests

@pytest.mark.integration
def test_send_real_request(start2serve):
    with open("tests/rocky.wav", "rb") as f:
        audio_bytes = f.read()

    res = requests.post(
        "http://localhost:8888/predictions/whisper_base", 
        files={"data": audio_bytes})
    assert res.status_code == 200
    transcription =  res.text.lower()
    assert "eu vou dizer uma coisa que você já sabe" in transcription
    assert "o mundo não é um grande arcoírus." in transcription
