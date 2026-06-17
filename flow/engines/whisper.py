"""faster-whisper engine (CTranslate2, CPU)."""

import logging

import numpy as np

from flow.engines import Transcriber

_SAMPLE_RATE = 16000
_MIN_SECONDS = 0.25

# Quiet the noisy download/progress loggers; warnings and errors still show.
for _name in ("faster_whisper", "huggingface_hub", "ctranslate2"):
    logging.getLogger(_name).setLevel(logging.WARNING)


class WhisperTranscriber(Transcriber):
    """Wraps a faster-whisper model for English transcription on CPU."""

    name = "whisper"

    def __init__(
        self,
        model_name: str = "base.en",
        compute_type: str = "int8",
        beam_size: int = 1,
    ) -> None:
        self.model_name = model_name
        self.compute_type = compute_type
        self.beam_size = beam_size
        self._model = None

    @property
    def label(self) -> str:
        return f"faster-whisper ({self.model_name})"

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model_name, device="cpu", compute_type=self.compute_type
        )

    def transcribe(self, audio: np.ndarray) -> str:
        if len(audio) < _SAMPLE_RATE * _MIN_SECONDS:
            return ""
        self.load()
        # vad_filter skips non-speech; condition_on_previous_text=False avoids
        # repetition spirals and their expensive temperature re-decodes.
        segments, _info = self._model.transcribe(
            audio,
            language="en",
            beam_size=self.beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return "".join(segment.text for segment in segments).strip()

    def unload(self) -> None:
        self._model = None
