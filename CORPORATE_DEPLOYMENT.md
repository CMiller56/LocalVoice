# Corporate / Constrained Environment Deployment Guide

This document is specifically for teams operating in large, regulated, air-gapped, or infrastructure-constrained environments (e.g. locked-down AWS, on-prem with old GPU drivers, massive existing dependency trees, etc.).

## Core Philosophy

The `LocalVoice` modules (`VoiceTranscriber` and `VoiceSynthesizer`) are designed to be **clean and reliable** when used correctly. However, they depend on heavy runtimes:

- `faster-whisper` → pulls in PyTorch + CTranslate2
- `piper-tts` → pulls in ONNX runtime

These two packages are the primary source of large environment sizes (often several GB). The rest of the LocalVoice code is intentionally lightweight.

**You will not solve a 5 GB+ or 97,000+ dependency problem by changing these two modules.** The bulk of the bloat almost always comes from the consuming application.

Our job is to make the voice components as **predictable and controllable** as possible.

---

## 1. Recommended Installation Strategy (Conda + CUDA Pinning)

### Step 1: Pin PyTorch / CUDA **first** (Critical)

Never let `faster-whisper` or `piper-tts` pull in their own PyTorch/ONNX versions. Always install the CUDA toolkit you are allowed to use **before** installing the voice packages.

Example for CUDA 12.1 (adjust to whatever your drivers support):

```bash
conda create -n voice python=3.11 -y
conda activate voice

# Pin the CUDA version you are allowed to use
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Now install the voice packages
pip install faster-whisper piper-tts numpy
```

For `sounddevice` on conda (recommended):

```bash
conda install python-sounddevice -c conda-forge
```

### Step 2: Pre-download models (Air-gapped / Reproducible)

**Never rely on auto-download in production or constrained environments.**

#### For Speech-to-Text (Whisper)

```bash
# Example: download small model once
python -c "
from faster_whisper import WhisperModel
model = WhisperModel('small', download_root='/opt/models/whisper')
print('Downloaded to /opt/models/whisper')
"
```

Then load with:

```python
transcriber = VoiceTranscriber(
    model_size="small",
    download_root="/opt/models/whisper"
)
```

#### For Text-to-Speech (Piper)

1. Go to: https://huggingface.co/rhasspy/piper-voices
2. Download the voice you want (e.g. `en_US-kathleen-medium`).
3. You need two files:
   - `model.onnx`
   - `model.onnx.json`

Recommended layout:

```
/opt/models/
├── whisper/
│   ├── small/
│   └── ...
└── piper/
    ├── en_US-kathleen-medium/
    │   ├── model.onnx
    │   └── model.onnx.json
    └── en_US-amy-medium/
        └── ...
```

Then load with:

```python
synthesizer = VoiceSynthesizer(
    model_dir="/opt/models/piper/en_US-kathleen-medium"
)
```

---

## 2. CUDA Version Control

Both `faster-whisper` (via PyTorch) and `piper-tts` (via ONNX) are sensitive to CUDA versions.

**Best practice:**

1. Decide on the CUDA version your infrastructure actually supports.
2. Install PyTorch with that exact CUDA version **before** installing `faster-whisper` or `piper-tts`.
3. Document the pinned versions in your environment specification (e.g. `environment.yml` or `requirements.txt` + comments).

Example `environment.yml` snippet:

```yaml
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pytorch==2.4.0
  - pytorch-cuda=12.1
  - pip
  - pip:
      - faster-whisper>=1.0.0
      - piper-tts>=1.2.0
      - numpy>=1.24
```

---

## 3. Environment Size Reduction Tips

- Install PyTorch from conda with the specific CUDA version (much better than pip in many cases).
- Avoid installing `torch` twice (once from conda, once pulled by pip packages).
- Do **not** install development extras or optional heavy packages unless needed.
- Consider using a minimal base image + explicit model mounting for containers.

Typical size breakdown (approximate):

- PyTorch + CUDA 12.1 runtime: ~2.5–3.5 GB
- CTranslate2 + faster-whisper models: varies with model size
- Piper + ONNX runtime: ~200–400 MB + voice models

The voice models themselves are usually the smaller part once the runtimes are installed.

---

## 4. Class Usage Recommendations for Constrained Environments

### VoiceTranscriber

```python
transcriber = VoiceTranscriber(
    model_size="small",                    # or "base", "medium", etc.
    download_root="/opt/models/whisper",   # pre-downloaded location
    device="cuda",                         # or "cpu"
    compute_type="int8_float16",           # good balance on many GPUs
)
```

### VoiceSynthesizer

```python
synthesizer = VoiceSynthesizer(
    model_dir="/opt/models/piper/en_US-kathleen-medium",
    use_cuda=False,                        # Piper CUDA support is limited
)
```

**Important:** Piper has much weaker CUDA support than PyTorch. In many corporate environments it is simpler and more reliable to run Piper on CPU.

---

## 5. Common Gotchas in Large Environments

- **"No module named 'piper.download'"** — This module is not part of the stable public API of the `piper-tts` package. Use `model_dir` instead.
- Mixing conda and pip PyTorch installations leads to mysterious CUDA errors.
- Some environments block outgoing HTTPS calls at runtime — pre-download everything.
- Very large dependency counts in the parent application can make debugging import order issues extremely painful.

---

## 6. Suggested Project Layout

```
/opt/models/                 # or /models/, /data/models/, etc.
├── whisper/
│   ├── tiny/
│   ├── base/
│   ├── small/
│   └── medium/
└── piper/
    ├── en_US-kathleen-medium/
    └── en_US-amy-medium/
        └── ...
```

Keep the model files completely separate from the application code. This makes upgrades, backups, and air-gapped transfers much easier.

---

## Summary

- Pin CUDA / PyTorch versions **before** installing voice packages.
- Pre-download all models and load them via `download_root` / `model_dir`.
- Treat `VoiceTranscriber` and `VoiceSynthesizer` as thin, well-documented wrappers around heavy runtimes.
- Accept that the majority of environment bloat in extreme cases (90k+ dependencies) comes from the host application, not these two modules.

If your infrastructure team can provide a stable PyTorch + CUDA base layer, the voice components become much more predictable and maintainable.

---

## Note on Backend Alternatives (as of 2026)

**Piper TTS** is the current backend for `VoiceSynthesizer`. It was selected for its excellent combination of small model sizes, strong CPU performance, mature offline tooling, and reliable behavior under constrained conditions.

**Kokoro-82M** (especially the ONNX variant) has emerged as a credible potential alternative. Early hands-on testing indicates it can deliver comparable or subjectively better naturalness while remaining relatively compact. 

If environment size, dependency count, or CUDA/driver constraints become significantly more pressing in the future, migrating the `VoiceSynthesizer` backend to Kokoro (or another strong ONNX-based model) is considered a viable trade-off. The public API of the class is designed to allow such a backend swap with limited disruption.

For the time being, Piper remains the default and recommended backend.