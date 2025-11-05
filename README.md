# 🧠 LLMRI

*A local-first visualization tool for watching a large language model “think.”*  
You can chat with a model, capture its per-token activations, map them into pixels, and scrub through time like an f-MRI for neural nets.

---

## 1. Overview

**Purpose:**  
This tool lets you observe how hidden states evolve inside a transformer as it generates text. Each token produces a “frame,” built from activations across all layers. You can:

• Chat with a model in real time  
• Watch activations light up per token  
• Scrub a timeline of frames  
• Change how activation magnitude is mapped to color/luminance  
• Export sessions for offline analysis  

**Core idea:**  
Map model activations → 2D “tiles” → per-token “frames” → interactive video.

---

## 2. Modes of Operation

**A. Full Activation Mode (preferred)**  
• Uses a local Transformers backend loading weights directly from `.safetensors`  
• Captures hidden states, attention, MLP signals  
• Provides full heatmaps

**B. Chat-Only Mode (fallback)**  
• Uses LM Studio or Ollama via their OpenAI-compatible APIs  
• Displays token timeline but no internal activations (since these APIs don’t expose hidden states)

---

## 3. System Architecture

**Server (Python/FastAPI)**  
• Streams token-by-token generation over WebSocket  
• Extracts hidden states and projects them to 2D tiles  
• Maintains session data for replay or scrubbing  

**Client (HTML + JavaScript)**  
• Receives streamed frames  
• Draws tiled activation maps on a `<canvas>`  
• Shows tokens, slider, and color mode controls  

**Reducer (Torch)**  
• Fixed projection matrix per layer (random or PCA)  
• Normalizes and compresses activations  
• Outputs compact `[H × W]` tiles per layer  

---

## 4. Visual Encoding

**Layout:**  
Each layer → one tile → arranged in a grid.

**Color channels:**  
• Luminance → normalized activation magnitude  
• Hue/Saturation → optional secondary info (signal type, sparsity, etc.)

**Defaults:**  
• Grayscale (magnitude only)  
• “Energy” mode (blue→red gradient)

---

## 5. Backends

| Backend | Activations? | API | Notes |
|----------|--------------|-----|------|
| **Transformers (local)** | ✅ full | direct weights | needs GPU VRAM |
| **LM Studio** | ❌ none | OpenAI API | chat only |
| **Ollama** | ❌ none | OpenAI/Ollama API | chat only |

If using LM Studio/Ollama, the viewer still runs, but the frames will be blank placeholders.

---

## 6. Installation

```bash
python3 -m venv ~/AI/venv
source ~/AI/venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn transformers torch torchvision pydantic
# optional
pip install accelerate einops safetensors sentencepiece
