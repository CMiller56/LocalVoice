# Voice Integration Guide (STT + TTS)

**Goal**: Help developers integrate the LocalVoice toolkit (VoiceTranscriber + VoiceSynthesizer) into larger Python applications.

This document covers patterns for using both speech-to-text and text-to-speech together.

---

## Core Classes

- `VoiceTranscriber` — Speech-to-Text (Whisper via faster-whisper)
- `VoiceSynthesizer` — Text-to-Speech (Piper)

Both classes are designed with similar APIs to make them easy to use together.

---

## Recommended Integration Patterns

See the individual module docstrings for detailed examples.

Common patterns include:
- FastAPI + WebSocket for real-time voice I/O
- GUI applications (Streamlit, desktop)
- Async wrappers using ThreadPoolExecutor
- Background workers for longer transcriptions or synthesis jobs

The key principle for both modules is: **Never block the main thread** with heavy audio work.

---

## Buffered TTS Playback

`VoiceSynthesizer.synthesize_stream()` supports a `buffer_ms` parameter. This is recommended over raw streaming for smoother audio output.

Example:
```python
for chunk, sr in synthesizer.synthesize_stream(text, buffer_ms=350):
    sd.play(chunk, sr)
    sd.wait()
```

---

## Offline Deployment

Both modules support fully manual model loading:
- STT: Use `download_root` + pre-cached Whisper models
- TTS: Use `model_dir` pointing to local Piper .onnx files

This makes the toolkit suitable for air-gapped and enterprise environments.

---

Further examples and advanced patterns will be expanded in future updates.