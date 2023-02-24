import os

import pytest

import whisper
from whisper.decoding import DecodingOptions


# @pytest.mark.parametrize('model_name', whisper.available_models())
# def test_transcribe(model_name: str):
#     model = whisper.load_model(model_name, download_root="./model_pt")
#     audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")

#     language = "en" if model_name.endswith(".en") else None
#     result = model.transcribe(audio_path, language=language, temperature=0.0)
#     assert result["language"] == "en"

#     transcription = result["text"].lower()
#     assert "my fellow americans" in transcription
#     assert "your country" in transcription
#     assert "do for you" in transcription


def test_transcribe_base_en():
    model_name = "base"
    model = whisper.load_model(model_name, download_root="./model_pt")
    audio_path = os.path.join(os.path.dirname(__file__), "rocky.wav")

    #language = "en" if model_name.endswith(".en") else None
    language = "pt"
    result = model.transcribe(audio_path, language=language)
    assert result["language"] == "pt"

    transcription = result["text"].lower()
    assert "eu vou dizer uma coisa que você já sabe" in transcription
    assert "o mundo não é um grande arcoírus." in transcription
