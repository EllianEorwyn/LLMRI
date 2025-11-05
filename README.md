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
pip install fastapi uvicorn transformers torch torchvision pydantic httpx pyyaml
# optional
pip install accelerate einops safetensors sentencepiece
```

---

## 7. Runtime configuration

Configuration is split between an optional `config.yaml` (server defaults) and the viewer UI (runtime overrides).

### `config.yaml`

Create a file next to `server.py` to set startup defaults. All keys are optional.

```yaml
server:
  host: 0.0.0.0
  port: 8008
  cors_origins: ["*"]

backend:
  kind: transformers        # transformers | lmstudio | ollama
  device: cuda:0
  dtype: bfloat16            # float16 | bfloat16 | float32

transformers:
  model_id: /models/Qwen3

lmstudio:
  base_url: http://localhost:1234/v1
  model: qwen:latest

ollama:
  base_url: http://localhost:11434
  model: llama3

reduction:
  tile_h: 32
  tile_w: 32
  projection: random         # random | pca
  rolling_window: 128
  pca_calibration_tokens: 256
```

* Tile geometry + projection method are loaded on boot. Switches at runtime keep the existing tiles until the next chat session.
* Leave `model_id` blank to defer loading until you apply a Transformers configuration from the UI.

### Viewer controls

Open `viewer.html` in a browser. Non-technical users can configure everything without touching the server:

* Server endpoints: editable REST + WebSocket URLs (point to any LAN host).
* Backend selector: Transformers (local weights), LM Studio, or Ollama.
* Transformers pane:
  * Model folder path on the server.
  * Device + dtype dropdowns.
  * “Pick folder” helper that opens the browser directory picker as a reminder to copy the correct server path.
* LM Studio pane: base URL + “Fetch models” button that calls `GET /models` (OpenAI-compatible) and fills the dropdown.
* Ollama pane: base URL + “Fetch models” button. It tries `/v1/models` first, then `/api/tags`.
* Apply button: POSTs to `/configure` and shows success/failure badges.
* Status badge: always reflects the active backend and whether activations are available.

---

## 8. Networking & LAN access

* The FastAPI server binds to `0.0.0.0` by default so any device on your LAN can connect (`http://<server-ip>:8008`).
* CORS defaults to `*` so you can host the HTML viewer elsewhere on the LAN (tighten this in `config.yaml` for production).
* The viewer only talks to the addresses you type—no cloud calls.
* Use a reverse proxy + auth if you plan to expose the service beyond your local network.

---

## 9. Feature roadmap highlights

* Runtime backend switching (`/configure`) without restarting the server.
* Transformers mode streams activation tiles (one per layer) in sync with tokens.
* LM Studio + Ollama modes reuse the same UI but stream chat-only timelines.
* Session API: `/reset/{conversation_id}` clears memory, `/frames/{conversation_id}` retrieves stored frames.
* Viewer fallback: blank canvases when activations are unavailable, timeline always updates.
* Robust error messages for bad paths, missing models, or unreachable APIs.
