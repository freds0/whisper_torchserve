import pytest

from handler import WhisperHandler
from tests.utils import MockContext


@pytest.fixture()
def serve_context():
    context = MockContext(
        model_name="base_whisper",
        model_type="base",
        model_dir="model_pt",
    )
    return context


def initialize(serve_context):
    handler = WhisperHandler()
    handler.initialize(serve_context)

    return handler


def test_handle(serve_context):
    context = serve_context
    handler = initialize(serve_context)
    with open("tests/rocky.wav", "rb") as f:
        audio_bytes = f.read()
    test_data = [{"data": audio_bytes}]
    results = handler.handle(test_data, context)

    assert len(results) == 1

    transcription = results[0].lower()
    assert "eu vou dizer uma coisa que você já sabe" in transcription
    assert "o mundo não é um grande arcoírus." in transcription


def test_handle_batch(serve_context):
    context = serve_context
    handler = initialize(serve_context)
    with open("tests/rocky.wav", "rb") as f:
        audio_bytes = f.read()

    test_data = [{"data": audio_bytes}] * 2
    results = handler.handle(test_data, context)
    assert len(results) == 2
    assert "eu vou dizer uma coisa que você já sabe" in results[1].lower()
