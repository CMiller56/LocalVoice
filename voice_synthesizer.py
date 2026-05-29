"""
Self-Hosted Text-to-Speech (TTS) Module using Piper
====================================================

This is the companion module to VoiceTranscriber. It provides a clean,
offline-capable text-to-speech system designed for easy integration into
larger Python applications.

Primary backend: Piper TTS (fast, high quality, excellent for offline use).

INSTALLATION
------------
    pip install piper-tts sounddevice numpy

    # No system ffmpeg is required for Piper (unlike some other TTS systems).

AUTO-DOWNLOAD BEHAVIOR
----------------------
By default, the module will automatically download a recommended high-quality
female voice on first use (if no local model is provided). This is convenient
for development.

For fully offline / air-gapped deployments, you can:
1. Pre-download the model files once.
2. Point the module at the local directory using `model_dir=...`

MANUAL / OFFLINE LOADING
------------------------
Pass `model_dir` pointing to a folder containing:
    - model.onnx
    - model.onnx.json

Example:
    synthesizer = VoiceSynthesizer(model_dir="/path/to/voices/kathleen")

RECOMMENDED VOICES (Female, warmer/contralto-leaning)
-----------------------------------------------------
- en_US-kathleen-medium   (good default - warm, mature female)
- en_US-amy-medium        (clear and pleasant)
- en_US-jenny_dioco-medium (natural, less bright)

You can change the default voice by passing `voice="en_US-amy-medium"` etc.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Generator, Union, Tuple

import numpy as np

try:
    from piper import PiperVoice
    from piper.download import ensure_voice_exists, get_voices
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


# Default recommended voice (warm female, contralto-leaning)
DEFAULT_VOICE = "en_US-kathleen-medium"


@dataclass
class SynthesisResult:
    """Structured result from text-to-speech synthesis."""
    audio: np.ndarray          # float32 audio data, mono, 16kHz or 22.05kHz depending on voice
    sample_rate: int
    duration: float            # in seconds
    voice: str
    text: str                  # original text that was synthesized


class VoiceSynthesizer:
    """
    Self-hosted text-to-speech using Piper TTS models.

    Designed to feel consistent with VoiceTranscriber for easy integration
    across STT and TTS in the same application.
    """

    def __init__(
        self,
        voice: str = "auto",
        model_dir: Optional[Union[str, Path]] = None,
        download_dir: Optional[Union[str, Path]] = None,
        use_cuda: bool = False,
    ):
        """
        Initialize the synthesizer.

        Args:
            voice: Name of the Piper voice to use (e.g. "en_US-kathleen-medium").
                   Use "auto" to let the module pick a good default female voice.
            model_dir: Path to a local directory containing a Piper model
                       (model.onnx + model.onnx.json). If provided, no network
                       access will be used.
            download_dir: Where to store downloaded voices. Defaults to
                          ~/.local/share/piper or similar.
            use_cuda: Whether to attempt GPU acceleration (limited support in Piper).
        """
        if not PIPER_AVAILABLE:
            raise ImportError(
                "piper-tts is not installed. "
                "Install with: pip install piper-tts"
            )

        self.download_dir = Path(download_dir) if download_dir else None
        self.use_cuda = use_cuda

        # Resolve voice
        if voice == "auto":
            voice = DEFAULT_VOICE

        self.voice_name = voice
        self.model_dir = Path(model_dir) if model_dir else None

        self._voice: Optional[PiperVoice] = None

        # Load or download the voice
        self._load_or_download_voice()

    def _load_or_download_voice(self):
        """Load a local model or auto-download the requested voice."""
        if self.model_dir:
            # Manual / offline mode - strict local loading
            model_path = self.model_dir / "model.onnx"
            config_path = self.model_dir / "model.onnx.json"

            if not model_path.exists() or not config_path.exists():
                raise FileNotFoundError(
                    f"Could not find Piper model files in {self.model_dir}. "
                    f"Expected 'model.onnx' and 'model.onnx.json'."
                )

            print(f"[VoiceSynthesizer] Loading local Piper model from {self.model_dir}...")
            self._voice = PiperVoice.load(
                str(model_path),
                config_path=str(config_path),
                use_cuda=self.use_cuda,
            )
            print("[VoiceSynthesizer] Local model loaded successfully (offline mode).")
            return

        # Auto-download mode (convenience)
        print(f"[VoiceSynthesizer] Ensuring voice '{self.voice_name}' is available...")
        start = time.time()

        # This will download if not present
        ensure_voice_exists(
            self.voice_name,
            data_dir=str(self.download_dir) if self.download_dir else None,
        )

        # Find the downloaded model
        voices_info = get_voices(
            str(self.download_dir) if self.download_dir else None
        )
        voice_info = voices_info[self.voice_name]

        model_path = voice_info["files"]["model.onnx"]["path"]
        config_path = voice_info["files"]["model.onnx.json"]["path"]

        self._voice = PiperVoice.load(
            model_path,
            config_path=config_path,
            use_cuda=self.use_cuda,
        )

        elapsed = time.time() - start
        print(f"[VoiceSynthesizer] Voice '{self.voice_name}' ready (loaded in {elapsed:.1f}s).")

    # ------------------------------------------------------------------ #
    # Core synthesis methods
    # ------------------------------------------------------------------ #

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
    ) -> SynthesisResult:
        """
        Synthesize text to speech and return the full audio.

        This is the simplest method and is good for shorter utterances.

        Args:
            text: The text to speak.
            speed: Speaking rate multiplier (1.0 = normal, 1.2 = faster, 0.8 = slower).

        Returns:
            SynthesisResult containing the audio array and metadata.
        """
        if not text or not text.strip():
            # Return silence for empty input
            return SynthesisResult(
                audio=np.zeros(1, dtype=np.float32),
                sample_rate=self._voice.config.sample_rate,
                duration=0.0,
                voice=self.voice_name,
                text=text,
            )

        start = time.time()
        audio = self._voice.synthesize(text, speed=speed)
        duration = len(audio) / self._voice.config.sample_rate

        return SynthesisResult(
            audio=audio.astype(np.float32),
            sample_rate=self._voice.config.sample_rate,
            duration=duration,
            voice=self.voice_name,
            text=text,
        )

    def synthesize_to_file(
        self,
        text: str,
        output_path: Union[str, Path],
        speed: float = 1.0,
    ) -> SynthesisResult:
        """
        Synthesize text and save directly to a WAV file.

        Convenient for generating audio assets or logging.
        """
        result = self.synthesize(text, speed=speed)

        import wave
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(result.sample_rate)
            # Convert float32 [-1, 1] to int16
            audio_int16 = (result.audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())

        return result

    def synthesize_stream(
        self,
        text: str,
        speed: float = 1.0,
        buffer_ms: int = 300,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        Synthesize text and yield audio in buffered chunks.

        This provides a good balance between responsiveness and smooth playback.
        Instead of yielding tiny fragments (which can sound choppy), we accumulate
        audio until we have at least `buffer_ms` worth before yielding.

        Recommended usage:
            for chunk, sr in synthesizer.synthesize_stream(text, buffer_ms=350):
                sd.play(chunk, sr)
                sd.wait()

        Args:
            text: Text to synthesize.
            speed: Speaking rate.
            buffer_ms: Minimum buffer size in milliseconds before yielding a chunk.
                       Higher values = smoother audio, slightly more latency.
                       200-400ms is usually a good range.

        Yields:
            (audio_chunk, sample_rate) tuples.
        """
        if not text or not text.strip():
            return

        sample_rate = self._voice.config.sample_rate
        buffer_samples = int((buffer_ms / 1000.0) * sample_rate)

        buffer = np.array([], dtype=np.float32)

        # Piper's synthesize_stream yields small chunks
        for audio_chunk in self._voice.synthesize_stream(text, speed=speed):
            buffer = np.concatenate([buffer, audio_chunk])

            while len(buffer) >= buffer_samples:
                yield buffer[:buffer_samples], sample_rate
                buffer = buffer[buffer_samples:]

        # Yield any remaining audio
        if len(buffer) > 0:
            yield buffer, sample_rate

    # ------------------------------------------------------------------ #
    # Convenience methods
    # ------------------------------------------------------------------ #

    def speak(
        self,
        text: str,
        speed: float = 1.0,
        blocking: bool = True,
    ):
        """
        Synthesize and immediately play the audio using sounddevice.

        Convenience method for quick testing and simple applications.
        Requires `sounddevice` to be installed.
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError("sounddevice is required for the .speak() method.")

        result = self.synthesize(text, speed=speed)
        sd.play(result.audio, result.sample_rate)
        if blocking:
            sd.wait()

    def get_voice_info(self) -> dict:
        """Return basic information about the loaded voice."""
        return {
            "voice": self.voice_name,
            "sample_rate": self._voice.config.sample_rate if self._voice else None,
            "model_loaded": self._voice is not None,
            "using_local_model": self.model_dir is not None,
        }

    @staticmethod
    def list_available_voices() -> list[str]:
        """Return a list of common high-quality Piper voices (female first)."""
        return [
            "en_US-kathleen-medium",
            "en_US-amy-medium",
            "en_US-jenny_dioco-medium",
            "en_US-ljspeech-medium",
            "en_GB-jenny_dioco-medium",
        ]


# ---------------------------------------------------------------------- #
# Quick self-test
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    print("=" * 60)
    print("VoiceSynthesizer - Quick Test")
    print("=" * 60)

    # This will auto-download a voice on first run
    synthesizer = VoiceSynthesizer(voice="auto")

    print("\nVoice info:", synthesizer.get_voice_info())

    test_text = "Hello. This is a test of the local voice synthesizer."

    print(f"\nSynthesizing: \"{test_text}\"")
    result = synthesizer.synthesize(test_text)
    print(f"Generated {result.duration:.2f} seconds of audio at {result.sample_rate} Hz.")

    # Optional: Play it
    # synthesizer.speak(test_text)

    print("\nTest complete. Use synthesize_stream() for real-time playback with buffering.")