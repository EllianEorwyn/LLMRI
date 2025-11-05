"""FastAPI server for the LLMRI activation viewer."""
from __future__ import annotations

import asyncio
import json
import logging
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is an optional dependency
    yaml = None


LOGGER = logging.getLogger("llmri.server")
logging.basicConfig(level=logging.INFO)


###############################################################################
# Configuration loading
###############################################################################
DEFAULT_CONFIG: Dict[str, Any] = {
    "server": {"host": "0.0.0.0", "port": 8008, "cors_origins": ["*"]},
    "backend": {"kind": "transformers", "device": "cuda:0", "dtype": "bfloat16"},
    "transformers": {"model_id": ""},
    "lmstudio": {"base_url": "http://localhost:1234/v1", "model": ""},
    "ollama": {"base_url": "http://localhost:11434", "model": ""},
    "reduction": {
        "tile_h": 32,
        "tile_w": 32,
        "projection": "random",
        "rolling_window": 128,
        "pca_calibration_tokens": 256,
    },
    "ui": {"static_dir": ""},
}

CONFIG_PATH = Path("config.yaml")


def deep_update(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config() -> Dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        if yaml is None:
            LOGGER.warning("PyYAML not installed, skipping config.yaml")
        else:
            with CONFIG_PATH.open("r", encoding="utf-8") as fh:
                file_cfg = yaml.safe_load(fh) or {}
            config = deep_update(config, file_cfg)
    return config


CONFIG = load_config()
SERVER_CFG = CONFIG.get("server", {})
BACKEND_CFG = CONFIG.get("backend", {})
TRANSFORMERS_CFG = CONFIG.get("transformers", {})
LMSTUDIO_CFG = CONFIG.get("lmstudio", {})
OLLAMA_CFG = CONFIG.get("ollama", {})
REDUCTION_CFG = CONFIG.get("reduction", {})

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


###############################################################################
# FastAPI setup
###############################################################################
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=SERVER_CFG.get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


###############################################################################
# Runtime state
###############################################################################
class RollingNorm:
    """Rolling window magnitude normalizer."""

    def __init__(self, cap: int, n_layers: int, m_pixels: int, eps: float = 1e-6) -> None:
        self.cap = cap
        self.eps = eps
        self.buffers: List[List[float]] = [[] for _ in range(n_layers)]

    def normalize(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        mag = float(x.abs().mean().item())
        buf = self.buffers[layer_idx]
        buf.append(mag)
        if len(buf) > self.cap:
            buf.pop(0)
        scale = max(max(buf), self.eps)
        return torch.clamp(x / scale, -1.0, 1.0)


class TransformerRuntime:
    def __init__(self) -> None:
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device: str = BACKEND_CFG.get("device", "cpu")
        self.dtype_str: str = BACKEND_CFG.get("dtype", "float32")
        self.dtype = DTYPE_MAP.get(self.dtype_str, torch.float32)
        self.tile_h: int = int(REDUCTION_CFG.get("tile_h", 32))
        self.tile_w: int = int(REDUCTION_CFG.get("tile_w", 32))
        self.rolling_window: int = int(REDUCTION_CFG.get("rolling_window", 128))
        self.projection_kind: str = str(REDUCTION_CFG.get("projection", "random"))
        self.pca_tokens: int = int(REDUCTION_CFG.get("pca_calibration_tokens", 0))
        self.projections: List[torch.Tensor] = []
        self.normer: Optional[RollingNorm] = None
        self.n_layers: int = 0
        self.d_model: int = 0

    def reset(self) -> None:
        self.model = None
        self.tokenizer = None
        self.projections = []
        self.normer = None
        self.n_layers = 0
        self.d_model = 0


TRANSFORMERS_STATE = TransformerRuntime()

ACTIVE_BACKEND: Dict[str, Any] = {
    "kind": BACKEND_CFG.get("kind", "transformers"),
    "transformers": {
        "model_id": TRANSFORMERS_CFG.get("model_id", ""),
        "device": BACKEND_CFG.get("device", "cpu"),
        "dtype": BACKEND_CFG.get("dtype", "float32"),
    },
    "lmstudio": {
        "base_url": LMSTUDIO_CFG.get("base_url", "http://localhost:1234/v1"),
        "model": LMSTUDIO_CFG.get("model", ""),
    },
    "ollama": {
        "base_url": OLLAMA_CFG.get("base_url", "http://localhost:11434"),
        "model": OLLAMA_CFG.get("model", ""),
    },
}

backend_lock = asyncio.Lock()
SESSIONS: Dict[str, Dict[str, List[Any]]] = {}


###############################################################################
# Utilities
###############################################################################
def torch_dtype_from_string(name: str) -> torch.dtype:
    if name not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{name}'. Choose from {list(DTYPE_MAP)}")
    return DTYPE_MAP[name]


def ensure_transformers_ready() -> None:
    if TRANSFORMERS_STATE.model is None or TRANSFORMERS_STATE.tokenizer is None:
        raise RuntimeError("Transformers backend is not loaded. Configure a model first.")


def current_status() -> Dict[str, Any]:
    has_activations = ACTIVE_BACKEND["kind"] == "transformers" and TRANSFORMERS_STATE.model is not None
    return {
        "backend": ACTIVE_BACKEND["kind"],
        "activations": has_activations,
        "tile_h": TRANSFORMERS_STATE.tile_h,
        "tile_w": TRANSFORMERS_STATE.tile_w,
    }


def empty_frame(signal: str) -> Dict[str, Any]:
    return {
        "tiles": [],
        "signal": signal,
        "tile_h": TRANSFORMERS_STATE.tile_h,
        "tile_w": TRANSFORMERS_STATE.tile_w,
    }


def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative <= top_p
    mask[..., 0] = True
    keep = torch.zeros_like(probs, dtype=torch.bool)
    keep.scatter_(dim=-1, index=idx, src=mask)
    filtered = torch.where(keep, probs, torch.zeros_like(probs))
    total = filtered.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return filtered / total


###############################################################################
# Projection builders
###############################################################################
def build_random_projections(
    d_model: int, m_pixels: int, n_layers: int, device: str, dtype: torch.dtype
) -> List[torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(42)
    mats = [
        torch.randn(d_model, m_pixels, generator=generator, device=device, dtype=dtype)
        / math.sqrt(d_model)
        for _ in range(n_layers)
    ]
    return mats


def build_pca_projections(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    dtype: torch.dtype,
    n_layers: int,
    m_pixels: int,
    calibration_tokens: int,
    d_model: int,
) -> List[torch.Tensor]:
    if calibration_tokens <= 0:
        LOGGER.warning("pca projection requested but calibration_tokens <= 0, falling back to random")
        return build_random_projections(d_model, m_pixels, n_layers, device, dtype)

    sample_text = (
        "The quick brown fox jumps over the lazy dog. This sentence is repeated to gather activations. "
        * max(1, calibration_tokens // 16)
    )
    encoded = tokenizer(sample_text, return_tensors="pt")
    input_ids = encoded["input_ids"][:, :calibration_tokens]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask[:, : input_ids.shape[1]]

    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states = outputs.hidden_states[1:]
    projections: List[torch.Tensor] = []
    q_target = min(m_pixels, input_ids.shape[1], hidden_states[0].shape[-1])
    for layer_hidden in hidden_states:
        data = layer_hidden[0].float()
        centered = data - data.mean(dim=0, keepdim=True)
        q = max(1, min(q_target, centered.shape[0], centered.shape[1]))
        if q == 1:
            basis = centered.mean(dim=0, keepdim=True).T
        else:
            try:
                _, _, v = torch.pca_lowrank(centered, q=q)
                basis = v[:, :q]
            except RuntimeError as exc:  # fallback if PCA fails
                LOGGER.warning("PCA failed for layer %s (%s), using random", len(projections), exc)
                basis = torch.randn(centered.shape[1], q)
        if basis.shape[1] < m_pixels:
            extra = torch.randn(centered.shape[1], m_pixels - basis.shape[1]) / math.sqrt(centered.shape[1])
            basis = torch.cat([basis, extra], dim=1)
        projections.append(basis[:, :m_pixels].to(device=device, dtype=dtype))
    return projections


###############################################################################
# Backend loading
###############################################################################
def load_transformers_backend(cfg: Dict[str, Any]) -> None:
    model_id = cfg.get("model_id") or TRANSFORMERS_CFG.get("model_id")
    if not model_id:
        raise ValueError("transformers.model_id must be provided")

    device = cfg.get("device") or ACTIVE_BACKEND["transformers"].get("device") or "cpu"
    dtype_name = cfg.get("dtype") or ACTIVE_BACKEND["transformers"].get("dtype") or "float32"
    dtype = torch_dtype_from_string(dtype_name)

    local_files_only = Path(model_id).exists()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()
    if not model.config.output_hidden_states:
        model.config.output_hidden_states = True

    # Inspect dimensions
    with torch.no_grad():
        sample = tokenizer("hello", return_tensors="pt")
        sample = {k: v.to(device) for k, v in sample.items()}
        out = model(**sample, output_hidden_states=True)
        n_layers = len(out.hidden_states) - 1
        d_model = out.hidden_states[-1].shape[-1]

    TRANSFORMERS_STATE.reset()
    TRANSFORMERS_STATE.model = model
    TRANSFORMERS_STATE.tokenizer = tokenizer
    TRANSFORMERS_STATE.device = device
    TRANSFORMERS_STATE.dtype_str = dtype_name
    TRANSFORMERS_STATE.dtype = dtype
    TRANSFORMERS_STATE.n_layers = n_layers
    TRANSFORMERS_STATE.d_model = d_model

    m_pixels = TRANSFORMERS_STATE.tile_h * TRANSFORMERS_STATE.tile_w
    if TRANSFORMERS_STATE.projection_kind == "pca":
        projections = build_pca_projections(
            model,
            tokenizer,
            device,
            dtype,
            n_layers,
            m_pixels,
            TRANSFORMERS_STATE.pca_tokens,
            d_model,
        )
    else:
        projections = build_random_projections(d_model, m_pixels, n_layers, device, dtype)

    TRANSFORMERS_STATE.projections = projections
    TRANSFORMERS_STATE.normer = RollingNorm(TRANSFORMERS_STATE.rolling_window, n_layers, m_pixels)

    LOGGER.info(
        "Loaded transformers backend %s on %s (%s) with %d layers",
        model_id,
        device,
        dtype_name,
        n_layers,
    )


###############################################################################
# API models
###############################################################################
class ConfigureRequest(BaseModel):
    kind: str
    transformers: Optional[Dict[str, Any]] = None
    lmstudio: Optional[Dict[str, Any]] = None
    ollama: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    conversation_id: str
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    signal: str = "residual"


###############################################################################
# Routes
###############################################################################
@app.get("/")
def root() -> Dict[str, Any]:
    info = current_status()
    info.update({
        "ok": True,
        "transformers_model": ACTIVE_BACKEND["transformers"].get("model_id", ""),
    })
    return info


@app.get("/frames/{conversation_id}")
def get_frames(conversation_id: str):
    sess = SESSIONS.get(conversation_id, {"frames": [], "tokens": []})
    return JSONResponse({"frames": sess.get("frames", []), "tokens": sess.get("tokens", [])})


@app.post("/reset/{conversation_id}")
def reset(conversation_id: str):
    SESSIONS[conversation_id] = {"frames": [], "tokens": []}
    return {"ok": True}


@app.post("/configure")
async def configure(req: ConfigureRequest):
    async with backend_lock:
        try:
            ACTIVE_BACKEND["kind"] = req.kind
            if req.transformers is not None:
                ACTIVE_BACKEND["transformers"].update(req.transformers)
            if req.lmstudio is not None:
                ACTIVE_BACKEND["lmstudio"].update(req.lmstudio)
            if req.ollama is not None:
                ACTIVE_BACKEND["ollama"].update(req.ollama)

            if req.kind == "transformers":
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, load_transformers_backend, ACTIVE_BACKEND["transformers"])
            LOGGER.info("Backend switched to %s", req.kind)
        except Exception as exc:  # pragma: no cover - runtime errors propagate to client
            LOGGER.exception("Failed to configure backend")
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    return {"ok": True, "kind": ACTIVE_BACKEND["kind"]}


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    try:
        await ws.send_text(json.dumps({"type": "server", "data": current_status()}))
        while True:
            raw = await ws.receive_text()
            payload = json.loads(raw)
            if payload.get("type") != "chat":
                continue
            req = ChatRequest(**payload["data"])
            try:
                if ACTIVE_BACKEND["kind"] == "transformers":
                    await run_chat_transformers(req, ws)
                elif ACTIVE_BACKEND["kind"] == "lmstudio":
                    await run_chat_openai_compatible(req, ws, ACTIVE_BACKEND["lmstudio"], "lmstudio")
                elif ACTIVE_BACKEND["kind"] == "ollama":
                    try:
                        await run_chat_openai_compatible(req, ws, ACTIVE_BACKEND["ollama"], "ollama")
                    except Exception:
                        await run_chat_ollama_native(req, ws, ACTIVE_BACKEND["ollama"])
                else:
                    raise RuntimeError(f"Unknown backend {ACTIVE_BACKEND['kind']}")
            except Exception as exc:
                await ws.send_text(json.dumps({"type": "error", "error": str(exc)}))
    except WebSocketDisconnect:
        return
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("WebSocket failure")
        try:
            await ws.send_text(json.dumps({"type": "error", "error": str(exc)}))
        except Exception:
            pass


###############################################################################
# Chat runners
###############################################################################
async def run_chat_transformers(req: ChatRequest, ws: WebSocket) -> None:
    ensure_transformers_ready()
    runtime = TRANSFORMERS_STATE
    tokenizer = runtime.tokenizer
    model = runtime.model
    assert tokenizer is not None and model is not None

    conversation = SESSIONS.setdefault(req.conversation_id, {"frames": [], "tokens": []})
    inputs = tokenizer(req.prompt, return_tensors="pt")
    inputs = {k: v.to(runtime.device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    generated = input_ids
    past_key_values = None

    for _ in range(req.max_new_tokens):
        outputs = model(
            input_ids=generated if past_key_values is None else generated[:, -1:],
            attention_mask=attention_mask if past_key_values is None else None,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        hidden_states = outputs.hidden_states[1:]

        last_tokens = [layer[:, -1, :].squeeze(0) for layer in hidden_states]
        tiles: List[List[List[int]]] = []
        for idx, vec in enumerate(last_tokens):
            projection = runtime.projections[idx]
            z = torch.matmul(vec.to(runtime.dtype), projection)
            normed = runtime.normer.normalize(idx, z)
            img = ((normed + 1.0) * 0.5).clamp(0.0, 1.0)
            pixels = (img * 255.0).round().to(torch.uint8).reshape(runtime.tile_h, runtime.tile_w)
            tiles.append(pixels.cpu().tolist())

        temperature = max(req.temperature, 1e-6)
        probs = torch.softmax(logits / temperature, dim=-1)
        if req.top_p < 1.0:
            probs = top_p_filter(probs, req.top_p)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
        token_text = tokenizer.decode(next_token[0].tolist(), skip_special_tokens=False)

        frame = {
            "tiles": tiles,
            "signal": req.signal,
            "tile_h": runtime.tile_h,
            "tile_w": runtime.tile_w,
        }
        conversation["frames"].append(frame)
        conversation["tokens"].append(token_text)

        await ws.send_text(
            json.dumps(
                {
                    "type": "frame",
                    "data": {
                        "token": token_text,
                        "index": len(conversation["frames"]) - 1,
                        "frame": frame,
                    },
                }
            )
        )

        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break


async def run_chat_openai_compatible(req: ChatRequest, ws: WebSocket, cfg: Dict[str, Any], label: str) -> None:
    base_url = cfg.get("base_url", "").rstrip("/")
    model_name = cfg.get("model")
    if not base_url or not model_name:
        raise ValueError(f"{label} base_url and model must be configured before chatting")

    conversation = SESSIONS.setdefault(req.conversation_id, {"frames": [], "tokens": []})
    payload = {
        "model": model_name,
        "stream": True,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": req.max_new_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
    }
    url = f"{base_url}/chat/completions"
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload, headers={"Accept": "text/event-stream"}) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[len("data: "):]
                else:
                    continue
                if data.strip() == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if not delta:
                    continue
                frame = empty_frame(req.signal)
                conversation["frames"].append(frame)
                conversation["tokens"].append(delta)
                await ws.send_text(
                    json.dumps({"type": "frame", "data": {"token": delta, "index": len(conversation["frames"]) - 1, "frame": frame}})
                )


async def run_chat_ollama_native(req: ChatRequest, ws: WebSocket, cfg: Dict[str, Any]) -> None:
    base_url = cfg.get("base_url", "").rstrip("/")
    model_name = cfg.get("model")
    if not base_url or not model_name:
        raise ValueError("ollama base_url and model must be configured before chatting")

    conversation = SESSIONS.setdefault(req.conversation_id, {"frames": [], "tokens": []})
    payload = {
        "model": model_name,
        "prompt": req.prompt,
        "stream": True,
        "options": {
            "temperature": req.temperature,
            "top_p": req.top_p,
            "num_predict": req.max_new_tokens,
        },
    }
    url = f"{base_url}/api/generate"
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    frame = empty_frame(req.signal)
                    conversation["frames"].append(frame)
                    conversation["tokens"].append(token)
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "frame",
                                "data": {"token": token, "index": len(conversation["frames"]) - 1, "frame": frame},
                            }
                        )
                    )
                if chunk.get("done"):
                    break


###############################################################################
# Entrypoint
###############################################################################
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=SERVER_CFG.get("host", "0.0.0.0"), port=int(SERVER_CFG.get("port", 8008)))
