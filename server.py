# server.py
import os
import math
import json
import asyncio
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# server.py additions and changes

from fastapi import FastAPI, WebSocket
from pydantic import BaseModel


# Keep existing imports and model code for the Transformers backend as before

# Active backend config in memory
ACTIVE_BACKEND = {
    "kind": "transformers",
    "transformers": {"model_id": "", "device": "cuda:0", "dtype": "bfloat16"},
    "lmstudio": {"base_url": "http://localhost:1234/v1", "model": ""},
    "ollama": {"base_url": "http://localhost:11434", "model": ""},
}

class ConfigureReq(BaseModel):
    kind: str
    transformers: dict | None = None
    lmstudio: dict | None = None
    ollama: dict | None = None

@app.post("/configure")
def configure(req: ConfigureReq):
    global ACTIVE_BACKEND, model, tokenizer, projections, N_LAYERS, D_MODEL, normer
    ACTIVE_BACKEND["kind"] = req.kind
    if req.transformers: ACTIVE_BACKEND["transformers"] = req.transformers
    if req.lmstudio: ACTIVE_BACKEND["lmstudio"] = req.lmstudio
    if req.ollama: ACTIVE_BACKEND["ollama"] = req.ollama

    # If switching to Transformers, (re)load the model
    if req.kind == "transformers":
        mid = ACTIVE_BACKEND["transformers"]["model_id"]
        dev = ACTIVE_BACKEND["transformers"]["device"]
        dt = ACTIVE_BACKEND["transformers"]["dtype"]
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dt]
        tokenizer = AutoTokenizer.from_pretrained(mid, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch_dtype, device_map={"": dev})
        model.config.output_hidden_states = True
        with torch.no_grad():
            dummy = tokenizer("hello", return_tensors="pt").to(dev)
            out = model(**dummy, output_hidden_states=True)
            N_LAYERS = len(out.hidden_states) - 1
            D_MODEL = out.hidden_states[-1].shape[-1]
        # rebuild projections and normer
        g = torch.Generator(device=dev).manual_seed(42)
        TILE_H, TILE_W = 32, 32
        M_PIXELS = TILE_H * TILE_W
        projections = [torch.randn(D_MODEL, M_PIXELS, generator=g, device=dev, dtype=model.dtype) / (D_MODEL ** 0.5) for _ in range(N_LAYERS)]
        normer = RollingNorm(128, N_LAYERS, M_PIXELS)
    return {"ok": True, "kind": ACTIVE_BACKEND["kind"]}

# In your websocket handler, send a small status line to the client
@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    try:
        await ws.send_text(json.dumps({"type": "server", "data": f"backend {ACTIVE_BACKEND['kind']}"}))
        while True:
            msg = await ws.receive_text()
            cmd = json.loads(msg)
            if cmd.get("type") == "chat":
                req = ChatRequest(**cmd["data"])
                # route by backend
                if ACTIVE_BACKEND["kind"] == "transformers":
                    await run_chat_transformers(req, ws)
                elif ACTIVE_BACKEND["kind"] == "lmstudio":
                    await run_chat_openai_compatible(req, ws, ACTIVE_BACKEND["lmstudio"]["base_url"], ACTIVE_BACKEND["lmstudio"]["model"])
                elif ACTIVE_BACKEND["kind"] == "ollama":
                    # try OpenAI compatible first, else fall back to native
                    try:
                        await run_chat_openai_compatible(req, ws, ACTIVE_BACKEND["ollama"]["base_url"] + "/v1", ACTIVE_BACKEND["ollama"]["model"])
                    except Exception:
                        await run_chat_ollama_native(req, ws, ACTIVE_BACKEND["ollama"]["base_url"], ACTIVE_BACKEND["ollama"]["model"])
    except WebSocketDisconnect:
        pass

# ======== Config ========
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-4B-Instruct")  # adjust if needed
DEVICE = os.environ.get("DEVICE", "cuda:0")
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Tile geometry
TILE_H = 32
TILE_W = 32
M_PIXELS = TILE_H * TILE_W  # m = 1024
ROLLING_NORM = 128  # normalize over last N tokens

# Which signals we support in MVP
SIGNALS = ["residual"]  # can add "attn_probs", "mlp_pre", "mlp_post" later

# ======== Model load ========
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map={"": DEVICE},
)
model.eval()

# Ensure we can get hidden states
if not model.config.output_hidden_states:
    model.config.output_hidden_states = True

# ======== Random projections per layer ========
# We will learn layer dims after a dummy forward
def inspect_dims(prompt_ids: torch.Tensor) -> Tuple[int, int]:
    with torch.no_grad():
        out = model(input_ids=prompt_ids, use_cache=True, output_hidden_states=True)
    n_layers = len(out.hidden_states) - 1  # includes embedding at 0
    d = out.hidden_states[-1].shape[-1]
    return n_layers, d

with torch.no_grad():
    dummy = tokenizer("hello", return_tensors="pt").to(DEVICE)
    N_LAYERS, D_MODEL = inspect_dims(dummy["input_ids"])

# One projection matrix per layer
# P_l: d x m, fixed seed for reproducibility
g = torch.Generator(device=DEVICE).manual_seed(42)
projections = [
    torch.randn(D_MODEL, M_PIXELS, generator=g, device=DEVICE, dtype=DTYPE) / math.sqrt(D_MODEL)
    for _ in range(N_LAYERS)
]

# ======== Rolling normalization buffers ========
class RollingNorm:
    def __init__(self, cap: int, n_layers: int, m_pixels: int, eps: float = 1e-6):
        self.cap = cap
        self.eps = eps
        self.buffers = [[] for _ in range(n_layers)]  # list of recent magnitudes per layer

    def normalize(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        # x: [m_pixels] float tensor on device
        mag = x.abs().mean().item()
        buf = self.buffers[layer_idx]
        buf.append(mag)
        if len(buf) > self.cap:
            buf.pop(0)
        scale = max(max(buf), self.eps)
        return torch.clamp(x / scale, -1.0, 1.0)

normer = RollingNorm(ROLLING_NORM, N_LAYERS, M_PIXELS)

# ======== Session storage ========
# Stores frames per conversation_id
SESSIONS: Dict[str, Dict] = {}

class ChatRequest(BaseModel):
    conversation_id: str
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    signal: str = "residual"  # which signal to paint into luminance

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/")
def root():
    return JSONResponse({"ok": True, "model": MODEL_ID, "layers": N_LAYERS, "d": D_MODEL})

@app.get("/frames/{conversation_id}")
def get_frames(conversation_id: str):
    sess = SESSIONS.get(conversation_id)
    if not sess:
        return JSONResponse({"frames": []})
    return JSONResponse({"frames": sess.get("frames", []), "tokens": sess.get("tokens", [])})

@app.post("/reset/{conversation_id}")
def reset(conversation_id: str):
    SESSIONS[conversation_id] = {"frames": [], "tokens": []}
    return {"ok": True}

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            # Expect a JSON command
            cmd = json.loads(msg)
            if cmd.get("type") == "chat":
                req = ChatRequest(**cmd["data"])
                await run_chat(req, ws)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "error": str(e)}))
        except Exception:
            pass

# ======== Core generation with per-token capture ========
@torch.no_grad()
async def run_chat(req: ChatRequest, ws: WebSocket):
    conv_id = req.conversation_id
    if conv_id not in SESSIONS:
        SESSIONS[conv_id] = {"frames": [], "tokens": []}

    inputs = tokenizer(req.prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    attn_mask = inputs.get("attention_mask", None)

    past_key_values = None
    generated = input_ids
    max_new = req.max_new_tokens

    for step in range(max_new):
        out = model(
            input_ids=generated if past_key_values is None else generated[:, -1:],
            attention_mask=attn_mask if past_key_values is None else None,
            use_cache=True,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )
        logits = out.logits[:, -1, :]
        past_key_values = out.past_key_values
        # hidden_states: tuple len N_LAYERS+1, take 1..N for layers
        h_states = out.hidden_states[1:]  # list of [B, T, D]
        last_h = [h[:, -1, :] for h in h_states]  # each [B, D], B=1
        last_h = [x.squeeze(0) for x in last_h]   # [D]

        # Project per layer to pixels and normalize
        tiles = []
        for l, hvec in enumerate(last_h):
            # residual-only MVP; hvec is residual stream output at layer l
            z = hvec.to(DTYPE) @ projections[l]  # [M_PIXELS]
            z = normer.normalize(l, z)  # [-1, 1]
            # map to 0..255 luminance; store as list of ints
            z01 = (z + 1.0) * 0.5  # 0..1
            z255 = torch.clamp((z01 * 255.0).round(), 0, 255).to(torch.uint8)
            tiles.append(z255.reshape(TILE_H, TILE_W).tolist())

        # Pick next token
        probs = torch.softmax(logits / max(1e-6, req.temperature), dim=-1)
        if req.top_p < 1.0:
            probs = top_p_filter(probs, req.top_p)
        next_id = torch.multinomial(probs, num_samples=1)  # [1, 1]
        generated = torch.cat([generated, next_id], dim=1)

        token_str = tokenizer.decode(next_id[0].tolist(), skip_special_tokens=False)

        # Save frame
        frame = {"tiles": tiles, "signal": req.signal, "tile_h": TILE_H, "tile_w": TILE_W}
        SESSIONS[conv_id]["frames"].append(frame)
        SESSIONS[conv_id]["tokens"].append(token_str)

        # Stream to client
        await ws.send_text(json.dumps({
            "type": "frame",
            "data": {
                "token": token_str,
                "index": len(SESSIONS[conv_id]["frames"]) - 1,
                "frame": frame
            }
        }))

        # Stop if EOS
        if next_id.item() == tokenizer.eos_token_id:
            break

def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep smallest set of tokens with cumulative prob >= top_p, renormalize."""
    sorted_probs, idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    mask = cum <= top_p
    # ensure at least one
    mask[..., 0] = True
    keep = torch.zeros_like(probs, dtype=torch.bool)
    keep.scatter_(dim=-1, index=idx, src=mask)
    pruned = torch.where(keep, probs, torch.zeros_like(probs))
    pruned = pruned / pruned.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return pruned

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8008"))
    uvicorn.run(app, host="0.0.0.0", port=port)