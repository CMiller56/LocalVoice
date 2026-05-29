# LocalVoice Integration Guide

This guide helps developers (and LLMs) integrate the **LocalVoice** toolkit into larger Python applications.

**Core Modules:**
- `VoiceTranscriber` — High-quality offline Speech-to-Text (powered by faster-whisper)
- `VoiceSynthesizer` — Fast, high-quality offline Text-to-Speech (powered by Piper)

Both classes were deliberately designed with similar APIs and patterns so they integrate cleanly together.

---

## Design Philosophy & Consistency

The two classes follow the same principles:

| Aspect                    | VoiceTranscriber                  | VoiceSynthesizer                     |
|---------------------------|-----------------------------------|--------------------------------------|
| Primary goal              | Accurate offline transcription    | Snappy, pleasant offline synthesis   |
| Model loading             | `model_size` or manual path       | Auto-download or `model_dir`         |
| Result object             | `TranscriptionResult`             | `SynthesisResult`                    |
| Key integration method    | `transcribe_audio_data()`         | `synthesize()` + `synthesize_stream()` |
| Threading rule            | Never block the main thread       | Never block the main thread          |
| Offline support           | Excellent                         | Excellent (stronger manual mode)     |

**Golden Rule (applies to both):**
> Never call transcription or synthesis directly from the main thread of a GUI, async web framework, or event loop.

---

## Recommended Class Instantiation Patterns

### Option 1: Simple (Good for scripts & prototypes)

```python
from voice_transcriber import VoiceTranscriber
from voice_synthesizer import VoiceSynthesizer

transcriber = VoiceTranscriber(model_size="small")
synthesizer = VoiceSynthesizer()   # auto-downloads a good voice
```

### Option 2: Production / Config-driven (Recommended)

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class VoiceConfig:
    stt_model_size: str = "small"
    tts_voice: str = "auto"                    # or "en_US-kathleen-medium"
    stt_download_root: Path | None = None
    tts_model_dir: Path | None = None          # for fully offline TTS
    device: str = "auto"

class VoiceManager:
    def __init__(self, config: VoiceConfig):
        self.transcriber = VoiceTranscriber(
            model_size=config.stt_model_size,
            download_root=config.stt_download_root,
        )
        self.synthesizer = VoiceSynthesizer(
            voice=config.tts_voice,
            model_dir=config.tts_model_dir,
        )
```

This pattern makes offline deployment and testing much easier.

---

## Using STT + TTS Together

### Basic Voice Response Loop (Buffered)

```python
def voice_interaction_loop(transcriber, synthesizer):
    while True:
        # 1. Listen (STT)
        print("Listening...")
        stt_result = transcriber.record_until_silence(max_duration=30)
        
        if not stt_result.text.strip():
            continue
            
        print(f"You said: {stt_result.text}")

        # 2. Generate reply (your LLM / logic here)
        reply_text = process_with_llm(stt_result.text)

        # 3. Speak (TTS with buffering for smoothness)
        print(f"Replying: {reply_text}")
        for chunk, sr in synthesizer.synthesize_stream(reply_text, buffer_ms=350):
            sd.play(chunk, sr)
            sd.wait()
```

**Why `buffer_ms=300–400`?**  
Pure streaming from Piper can sound choppy. A small buffer produces much more natural playback while still feeling responsive.

---

## Threading & Async Patterns

### FastAPI Example (Recommended)

```python
from fastapi import FastAPI, BackgroundTasks
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
voice_manager = VoiceManager(...)  # see above
executor = ThreadPoolExecutor(max_workers=2)

async def process_voice_input(audio_bytes: bytes):
    loop = asyncio.get_running_loop()
    
    # STT in background thread
    stt_result = await loop.run_in_executor(
        executor, 
        lambda: voice_manager.transcriber.transcribe_audio_data(...)
    )
    
    reply = await your_llm_call(stt_result.text)
    
    # TTS in background thread
    await loop.run_in_executor(
        executor,
        lambda: voice_manager.synthesizer.speak(reply)
    )
```

### Streamlit / GUI

Use `st.cache_resource` for the manager and always run synthesis/transcription via `executor.submit()` or `asyncio.to_thread`.

---

## Offline & Air-Gapped Deployment

### Fully Offline TTS (Strongly Recommended for Production)

```python
# Download once on a machine with internet
# Then copy the folder

synthesizer = VoiceSynthesizer(
    model_dir="/opt/voices/en_US-kathleen-medium"   # contains model.onnx + .json
)
```

### Fully Offline STT

```python
transcriber = VoiceTranscriber(
    model_size="small",
    download_root="/opt/whisper-models"
)
```

**Tip:** Pre-download models during CI/CD or deployment and bake them into your Docker image or installation package.

---

## Error Handling Recommendations

Wrap calls in both modules. Common issues:

- **STT**: No speech detected, very short audio, missing ffmpeg (for non-wav)
- **TTS**: Empty text, model files missing when using `model_dir`

Create wrapper functions:

```python
def safe_transcribe(audio, sr):
    try:
        return transcriber.transcribe_audio_data(audio, sr)
    except Exception as e:
        logger.error(f"STT failed: {e}")
        return None

def safe_synthesize(text):
    if not text or not text.strip():
        return
    try:
        return synthesizer.synthesize(text)
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        return None
```

---

## Testing Strategy

**Never run real models in unit tests.**

Create simple fakes:

```python
class FakeVoiceManager:
    def __init__(self):
        self.transcriber = FakeTranscriber()
        self.synthesizer = FakeSynthesizer()

class FakeTranscriber:
    def transcribe_audio_data(self, audio, sr, **kwargs):
        return TranscriptionResult(
            text="This is a test transcription.",
            language="en", language_probability=0.99,
            duration=2.3, model="fake", segments=None
        )

class FakeSynthesizer:
    def synthesize(self, text, **kwargs):
        return SynthesisResult(
            audio=np.zeros(16000, dtype=np.float32),
            sample_rate=16000, duration=1.0,
            voice="fake", text=text
        )
    
    def synthesize_stream(self, text, **kwargs):
        yield (np.zeros(8000, dtype=np.float32), 16000)
```

Use dependency injection so you can swap the real manager for the fake in tests.

---

## Quick Integration Checklist

- [ ] Instantiate `VoiceManager` (or equivalent) once at startup
- [ ] Never call STT/TTS from the main thread
- [ ] Use buffered TTS streaming (`buffer_ms` ≈ 300–400) for voice responses
- [ ] Support manual model paths for offline/air-gapped deployments
- [ ] Add proper error handling around both modules
- [ ] Write fakes for unit tests
- [ ] Set reasonable timeouts for long recordings or synthesis jobs
- [ ] Log detected language (STT) and voice used (TTS)

---

## When to Use Which Module

- Use `VoiceTranscriber` when you need to understand spoken input.
- Use `VoiceSynthesizer` when you need to speak back to the user.
- Use both together when building voice assistants, accessibility features, or hands-free interfaces.

The two modules were built to work as a pair.

---

**Questions worth asking before integrating:**
- Do you need real-time bidirectional voice, or is push-to-talk acceptable?
- What is your target latency for TTS playback?
- Are you deploying in air-gapped or low-connectivity environments?
- Will this run on CPU-only hardware?

This guide + the docstrings in both Python files should give another LLM enough context to produce a high-quality integration. Update this document as you discover new patterns in your specific application.