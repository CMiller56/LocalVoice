# LocalVoice

Self-hosted speech toolkit for Python applications.

**Goal**: Provide clean, reliable, fully offline speech-to-text and text-to-speech modules that are easy to integrate into larger Python projects.

---

## Features

- **Speech-to-Text (STT)**: Powered by faster-whisper (excellent accuracy + speed)
- **Text-to-Speech (TTS)**: Powered by Piper (fast, high quality, great for real-time use)
- Designed for easy integration (consistent APIs across STT and TTS)
- Strong support for offline / air-gapped environments
- Buffered streaming support in TTS for smooth playback
- Comprehensive integration documentation

---

## Modules

| Module                    | Purpose                          | Main Class            |
|---------------------------|----------------------------------|-----------------------|
| `voice_transcriber.py`    | Speech-to-Text (Whisper)         | `VoiceTranscriber`    |
| `voice_synthesizer.py`    | Text-to-Speech (Piper)           | `VoiceSynthesizer`    |

---

## Quick Start

### Installation

```bash
# For both STT + TTS
pip install faster-whisper piper-tts sounddevice numpy scipy

# Or install individually
pip install -r voice-requirements.txt
pip install -r voice-tts-requirements.txt
```

### Speech-to-Text

```python
from voice_transcriber import VoiceTranscriber

transcriber = VoiceTranscriber(model_size="small")
result = transcriber.transcribe_file("meeting.mp3")
print(result.text)
```

### Text-to-Speech

```python
from voice_synthesizer import VoiceSynthesizer

synthesizer = VoiceSynthesizer()  # Auto-downloads a good female voice
synthesizer.speak("Hello, this is a local voice.")

# Or get audio data
result = synthesizer.synthesize("This is buffered synthesis.")
```

See the individual files and `VOICE_INTEGRATION.md` for advanced usage.

---

## Offline / Air-Gapped Usage

Both modules are designed to work completely offline after initial setup:

- **STT**: Download your chosen Whisper model once and use `download_root`.
- **TTS**: Download a Piper voice once and load it with `model_dir="/path/to/voice"`.

**For corporate, constrained, or large environments**, see the dedicated guide:

→ [CORPORATE_DEPLOYMENT.md](./CORPORATE_DEPLOYMENT.md)

This covers CUDA version pinning, conda installation quirks, recommended directory layouts, and environment size management.

---

## Voice Quality Notes (TTS)

The default TTS voice (`en_US-kathleen-medium`) was chosen as a warm, mature female voice that many people find pleasant for longer interactions.

Other good options are listed in `voice_synthesizer.py`.

**Note:** While Piper is the current backend, Kokoro-82M (particularly the ONNX version) is documented as a potential future alternative if environment size or dependency constraints become more acute. See `CORPORATE_DEPLOYMENT.md` for details.

---

## Integration

Both modules follow a consistent design so they integrate cleanly together in the same application.

See [VOICE_INTEGRATION.md](./VOICE_INTEGRATION.md) for patterns covering:
- FastAPI + WebSocket
- GUI applications (Streamlit, Tkinter, etc.)
- Async wrappers
- Background workers
- Testing strategies
- Production hardening

---

## Repository Structure

```
LocalVoice/
├── voice_transcriber.py          # STT module
├── voice_synthesizer.py          # TTS module
├── CORPORATE_DEPLOYMENT.md       # Guidance for constrained environments
├── VOICE_INTEGRATION.md          # Detailed integration guide
├── voice-requirements.txt
├── voice-tts-requirements.txt
├── README.md
└── LICENSE
```

---

## License

MIT License (see LICENSE file).

---

## Contributing

This project focuses on practical, production-friendly offline speech capabilities. Contributions that improve integration experience, documentation, or reliability are very welcome.

---

## Related Projects

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Piper TTS](https://github.com/rhasspy/piper)

If you're building applications that need local voice I/O without relying on cloud services, this toolkit should give you a solid foundation.